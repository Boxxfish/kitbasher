from argparse import ArgumentParser
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import *
import gymnasium as gym
from kitbasher_rust import EngineWrapper, PyPlacedConfig
import torch_geometric  # type: ignore
from torch_geometric.data import Data, Batch  # type: ignore
from torch_geometric.nn.conv import GCNConv  # type: ignore
from torch_geometric.nn import DeepSetsAggregation  # type: ignore
from torch_geometric.nn import Sequential  # type: ignore
from torch_geometric.nn import aggr
from torch import Tensor
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch_geometric.utils  # type: ignore
import wandb
from tqdm import tqdm
from safetensors.torch import save_model, load_model

from kitbasher.algorithms.dqn import train_dqn
from kitbasher.algorithms.replay_buffer import ReplayBuffer
from kitbasher.env import ConstructionEnv
from kitbasher.pretraining import FeatureExtractor, Pretrained
from kitbasher.scorers import connect_scorer, connect_start, create_clip_scorer, single_start, volume_fill_scorer
from kitbasher.utils import create_directory, parse_args
from kitbasher.pretraining import ExpMeta as PretrainingExpMeta

_: Any
INF = 10**8

LABELS = [
    "sports car",
    "rowboat",
    "motorcycle",
    "tractor",
    "train",
    "helicopter",
    "plane",
    "spaceship",
    "skateboard",
    "hot air balloon",
]


# Hyperparameters
class Config(BaseModel):
    train_steps: int = 1  # Number of steps to step through during sampling.
    iterations: int = 100000  # Number of sample/train iterations.
    train_iters: int = 1  # Number of passes over the samples collected.
    train_batch_size: int = 256  # Minibatch size while training models.
    discount: float = 1.0  # Discount factor applied to rewards.
    q_epsilon: float = (
        1.0  # Epsilon for epsilon greedy strategy. This gets annealed over time.
    )
    eval_steps: int = 8  # Number of eval runs to average over.
    max_eval_steps: int = 300  # Max number of steps to take during each eval run.
    q_lr: float = 0.0001  # Learning rate of the q net.
    buffer_size: int = 10_000  # Number of elements that can be stored in the buffer.
    target_update: int = 500  # Number of iterations before updating Q target.
    save_every: int = 100
    max_steps: int = (
        64  # Maximum number of steps that can be performed in the environment.
    )
    score_fn: str = (
        "volume"  # Score function to use. Choices: "volume", "connect", "clip".
    )
    use_potential: bool = (
        False  # If true, the agent is rewarded by the change in scoring on each timestep. Otherwise, the reward is only given at the end.
    )
    process_type: str = (
        "gcn"  # Type of operation in the processing step. Choices: ["deep_set", "gcn", "self_attn", "independent"]
    )
    max_actions_per_step: int = (
        100  # The maximum number of placements the environment provides at each step.
    )
    prompt: str = "a lego "
    process_layers: int = 2  # The number of layers in the process step.
    tanh_logit: bool = (
        False  # Whether we should apply the tanh activation function on the q value.
    )
    eval_every: int = 100
    norm_min: float = 0.7
    norm_max: float = 1.2
    no_advantage: bool = False
    out_dir: str = "runs"
    use_mirror: bool = False
    fe_path: str = ""
    single_class: str = ""
    device: str = "cuda"


class ExpMeta(BaseModel):
    args: Config


class Lambda(nn.Module):
    """
    A parameterless module that just wraps a function.
    """

    def __init__(self, fn: Callable):
        nn.Module.__init__(self)
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class QNet(nn.Module):
    def __init__(
        self,
        num_parts: int,
        part_emb_size: int,
        num_steps: int,
        node_feature_dim: int,
        hidden_dim: int,
        process_type: str,
        tanh_logit: bool,
        no_advantage: bool,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        nn.Module.__init__(self)
        assert process_type in ["deep_set", "gcn", "self_attn", "independent"]

        # Part embeddings
        self.embeddings = nn.Parameter(torch.rand([num_parts, part_emb_size]))

        # Encode-process-encode architecture
        self.encode = nn.Linear(part_emb_size + node_feature_dim, hidden_dim)
        process_layers: List[Union[Tuple[nn.Module, str], nn.Module]] = []
        for _ in range(num_steps):
            if process_type == "deep_set":
                process_layers.append(
                    (
                        DeepSetsAggregation(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        "x, batch -> x",
                    )
                )
                process_layers.append(
                    (
                        Lambda(lambda x, batch: torch.index_select(x, 0, batch)),
                        "x, batch -> x",
                    )
                )
            if process_type == "gcn":
                process_layers.append(
                    (GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x")
                )
            if process_type == "independent":
                process_layers.append((nn.Linear(hidden_dim, hidden_dim), "x -> x"))
            process_layers.append(nn.ReLU())
        self.process = Sequential("x, edge_index, batch", process_layers)
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mean_aggr = aggr.MeanAggregation()

        self.use_advantage = not no_advantage
        if self.use_advantage:
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )
        self.tanh_logit = tanh_logit
        self.feature_extractor = feature_extractor

    def forward(self, data: Data):
        data = data.sort()
        x, edge_index, batch, part_ids = (
            data.x,
            data.edge_index,
            data.batch,
            data.part_ids,
        )
        if self.feature_extractor:
            x = self.feature_extractor(data)  # Shape: (num_nodes, hidden_dim)
        else:
            edge_index = torch_geometric.utils.add_self_loops(edge_index)[0]
            part_embs = self.embeddings.index_select(
                0, part_ids
            )  # Shape: (num_nodes, part_emb_dim)
            node_embs = torch.cat(
                [part_embs, x], 1
            )  # Shape: (num_nodes, node_dim + part_emb_dim)
            x = self.encode(node_embs)  # Shape: (num_nodes, hidden_dim)
        x = self.process(x, edge_index, batch)  # Shape: (num_nodes, hidden_dim)
        advantage = self.advantage(x)  # Shape: (num_nodes, 1)

        # If advantages are enabled, q values are a combination of value + advantage
        if self.use_advantage:
            advantage_mean = self.mean_aggr(advantage, batch)  # Shape: (num_batches, 1)
            advantage_mean = torch.gather(
                advantage_mean, 0, batch.unsqueeze(1)
            )  # Shape: (num_nodes, 1)
            value_x = self.mean_aggr(x, batch)  # Shape: (num_batches, hidden_dim)
            value = self.value(value_x)  # Shape: (num_batches, 1)
            value = torch.gather(value, 0, batch.unsqueeze(1))  # Shape: (num_nodes, 1)
            q_val = value + advantage - advantage_mean
        # If advantages are disabled, the advantages directly become values
        else:
            q_val = advantage

        if self.tanh_logit:
            q_val = torch.tanh(q_val)
        return q_val


def get_action(q_net: nn.Module, obs: Data, action_mask: Tensor) -> tuple[int, float]:
    q_vals = q_net(obs).squeeze(1)  # Shape: (num_nodes)
    q_vals = torch.masked_fill(q_vals, action_mask, -torch.inf)
    action = q_vals.argmax(0).item()
    q_val = q_vals.amax(0).item()
    return action, q_val


def process_obs(obs: Data) -> Batch:
    return Batch.from_data_list([obs])


def process_act_masks(obs: Data) -> Tensor:
    return obs.action_mask


if __name__ == "__main__":
    cfg: Config = parse_args(Config)
    device = torch.device(cfg.device)

    wandb.init(
        project="kitbasher",
        config=cfg.__dict__,
    )

    # Create out directory
    chkpt_path = create_directory(cfg.out_dir, ExpMeta(args=cfg))

    if cfg.score_fn == "volume":
        score_fn = volume_fill_scorer
        start_fn = single_start
    elif cfg.score_fn == "connect":
        score_fn = connect_scorer
        start_fn = connect_start
    elif cfg.score_fn == "clip":
        score_fn = create_clip_scorer()
        start_fn = single_start
    else:
        raise NotImplementedError(f"Invalid score function, got {cfg.score_fn}")
    labels = LABELS
    if cfg.single_class:
        labels = [cfg.single_class]
    prompts = [cfg.prompt + l for l in labels]
    env = ConstructionEnv(
        score_fn=score_fn,
        start_fn=start_fn,
        max_actions_per_step=cfg.max_actions_per_step,
        use_potential=cfg.use_potential,
        max_steps=cfg.max_steps,
        prompts=prompts,
        use_mirror=cfg.use_mirror,
    )
    test_env = ConstructionEnv(
        score_fn=score_fn,
        start_fn=start_fn,
        max_actions_per_step=cfg.max_actions_per_step,
        use_potential=cfg.use_potential,
        max_steps=cfg.max_steps,
        prompts=prompts,
        use_mirror=cfg.use_mirror,
    )

    # Initialize Q network
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Graph)
    assert isinstance(obs_space.node_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Discrete)
    feature_extractor = None
    if cfg.fe_path:
        meta_path = Path(cfg.fe_path).parent.parent / "meta.json"
        with open(meta_path, "r") as f:
            meta = PretrainingExpMeta.model_validate_json(f.read())
        clip_dim = 512
        pretrained = Pretrained(
            env.num_parts,
            meta.cfg.part_emb_size,
            meta.cfg.num_steps,
            obs_space.node_space.shape[0],
            64,
            clip_dim,
        )
        load_model(pretrained, cfg.fe_path)
        feature_extractor = pretrained.feature_extractor
    q_net = QNet(
        env.num_parts,
        32,
        cfg.process_layers,
        obs_space.node_space.shape[0],
        64,
        cfg.process_type,
        tanh_logit=cfg.tanh_logit,
        no_advantage=cfg.no_advantage,
        feature_extractor=feature_extractor,
    )
    q_net_target = copy.deepcopy(q_net)
    q_net_target.to(device)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=cfg.q_lr)

    # A replay buffer stores experience collected over all sampling runs
    buffer = ReplayBuffer(
        torch.Size((int(act_space.n),)),
        cfg.buffer_size,
    )

    obs_, info = env.reset()
    obs = process_obs(obs_)
    mask = process_act_masks(obs_)
    warmup_steps = int(cfg.buffer_size / cfg.train_steps)
    for step in tqdm(range(warmup_steps + cfg.iterations), position=0):
        percent_done = max((step - warmup_steps) / cfg.iterations, 0)

        # Collect experience
        with torch.no_grad():
            for _ in range(cfg.train_steps):
                if (
                    random.random() < cfg.q_epsilon * max(1.0 - percent_done, 0.05)
                    or not buffer.filled
                ):
                    action = random.choice(
                        [i for i, b in enumerate((~mask.bool()).tolist()) if b]
                    )
                else:
                    action, _ = get_action(q_net, obs, mask)
                obs_, reward, done, trunc, info_ = env.step(action)

                # Normalize reward if last step
                if (done or trunc) and not cfg.use_potential:
                    reward = (reward - cfg.norm_min) / (cfg.norm_max - cfg.norm_min)

                next_obs = process_obs(obs_)
                next_mask = process_act_masks(obs_)
                buffer.insert_step(
                    [obs.to_data_list()[0]],
                    [next_obs.to_data_list()[0]],
                    torch.tensor([action]),
                    [reward],
                    [done],
                )
                obs = next_obs
                mask = next_mask
                if done or trunc:
                    obs_, info = env.reset()
                    obs = process_obs(obs_)
                    mask = process_act_masks(obs_)

        # Train
        if buffer.filled:
            total_q_loss = train_dqn(
                q_net,
                q_net_target,
                q_opt,
                buffer,
                device,
                cfg.train_iters,
                cfg.train_batch_size,
                cfg.discount,
            )

            log_dict = {
                "avg_q_loss": total_q_loss / cfg.train_iters,
                "q_lr": q_opt.param_groups[-1]["lr"],
            }

            # Evaluate the network's performance after this training iteration.
            if step % cfg.eval_every == 0:
                with torch.no_grad():
                    reward_total = 0.0
                    pred_reward_total = 0.0
                    max_reward_total = -float("inf")
                    min_reward_total = float("inf")
                    obs_, _ = test_env.reset()
                    eval_obs = process_obs(obs_)
                    eval_mask = process_act_masks(obs_)
                    for _ in range(cfg.eval_steps):
                        steps_taken = 0
                        episode_reward = 0.0
                        for _ in range(cfg.max_eval_steps):
                            action, q_val = get_action(q_net, eval_obs, eval_mask)
                            pred_reward_total += q_val
                            obs_, reward, done, trunc, _ = test_env.step(action)
                            eval_obs = eval_obs = process_obs(obs_)
                            eval_mask = process_act_masks(obs_)
                            steps_taken += 1
                            reward_total += reward
                            episode_reward += reward
                            if done or trunc:
                                obs_, info = test_env.reset()
                                eval_obs = process_obs(obs_)
                                eval_mask = process_act_masks(obs_)
                                break
                        max_reward_total = max(episode_reward, max_reward_total)
                        min_reward_total = min(episode_reward, min_reward_total)
                log_dict.update(
                    {
                        "avg_eval_episode_reward": reward_total / cfg.eval_steps,
                        "eval_max_reward": max_reward_total,
                        "eval_min_reward": min_reward_total,
                        "avg_eval_episode_predicted_reward": pred_reward_total
                        / cfg.eval_steps,
                    }
                )

            wandb.log(log_dict)

            # Update Q target
            if step % cfg.target_update == 0:
                q_net_target.load_state_dict(q_net.state_dict())

            # Save checkpoint
            if step % cfg.save_every == 0:
                save_model(
                    q_net,
                    str(chkpt_path / f"q_net-{step}.safetensors"),
                )
                save_model(
                    q_net,
                    str(chkpt_path / f"q_net-latest.safetensors"),
                )
