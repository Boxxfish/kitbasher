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
from kitbasher.env import BLOCK_PARTS, ConstructionEnv
from kitbasher.pretraining import FeatureExtractor, Pretrained
from kitbasher.distributed_scorer import (
    get_scorer_fn,
)
from kitbasher.utils import create_directory, get_action, parse_args
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
    freeze_fe: bool = False
    freeze_fe_for: int = -1
    single_class: str = "" # This can be a comma separated list too
    single_class_selected: str = "" # This can be a comma separated list too
    distr_scorer: bool = False
    max_queued_items: int = 8
    num_render_workers: int = 2
    part_emb_size: int = 32
    hidden_dim: int = 64
    last_step_sample_bonus: float = 1.0 # How many times likely the final step will be sampled compared to previous steps
    last_step_sample_bonus_start: float = 1.0 # Last step sample bonus at the start; will anneal to `last_step_sample_bonus`
    last_step_sample_bonus_start_steps: int = 0 # Annealing time for last step sample bonus
    add_steps: bool = False
    use_traj_returns: bool = False
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
        freeze_fe: bool,
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
        if freeze_fe:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def freeze_fe(self, should_freeze: bool):
        for param in self.feature_extractor.parameters():
            param.requires_grad = not should_freeze

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


def process_obs(obs: Data) -> Batch:
    return Batch.from_data_list([obs])


def process_act_masks(obs: Data) -> Tensor:
    return obs.action_mask.bool()


if __name__ == "__main__":
    cfg: Config = parse_args(Config)
    device = torch.device(cfg.device)
    assert not (cfg.use_traj_returns and cfg.use_potential), "Trajectory returns can only be used sparsely"

    wandb.init(
        project="kitbasher",
        config=cfg.__dict__,
    )

    # Create out directory
    chkpt_path = create_directory(cfg.out_dir, ExpMeta(args=cfg))

    labels = LABELS
    if cfg.single_class:
        labels = [c.strip() for c in cfg.single_class.split(",")]
    selected_labels = list(range(len(labels)))
    if cfg.single_class_selected:
        selected_labels = [labels.index(c) for c in cfg.single_class_selected.split(",")]
    prompts = [cfg.prompt + l for l in labels]
    score_fn, eval_score_fn, start_fn, scorer_manager = get_scorer_fn(
        score_fn_name=cfg.score_fn,
        distr_scorer=cfg.distr_scorer,
        max_queued_items=cfg.max_queued_items,
        use_mirror=cfg.use_mirror,
        prompts=prompts,
        num_render_workers=cfg.num_render_workers,
        part_paths=[path.replace(".ron", ".glb") for path in BLOCK_PARTS],
        norm_min=cfg.norm_min,
        norm_max=cfg.norm_max,
        use_potential=cfg.use_potential,
    )
    with scorer_manager() as scorer:
        env = ConstructionEnv(
            score_fn=score_fn,
            start_fn=start_fn,
            max_actions_per_step=cfg.max_actions_per_step,
            use_potential=cfg.use_potential,
            max_steps=cfg.max_steps,
            prompts=prompts,
            selected_prompts=selected_labels,
            use_mirror=cfg.use_mirror,
            add_steps=cfg.add_steps,
        )
        test_env = ConstructionEnv(
            score_fn=eval_score_fn,
            start_fn=start_fn,
            max_actions_per_step=cfg.max_actions_per_step,
            use_potential=cfg.use_potential,
            max_steps=cfg.max_steps,
            prompts=prompts,
            selected_prompts=selected_labels,
            use_mirror=cfg.use_mirror,
            add_steps=cfg.add_steps,
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
                cfg.hidden_dim,
                clip_dim,
            )
            load_model(pretrained, cfg.fe_path)
            feature_extractor = pretrained.feature_extractor
        q_net = QNet(
            env.num_parts,
            cfg.part_emb_size,
            cfg.process_layers,
            obs_space.node_space.shape[0],
            cfg.hidden_dim,
            cfg.process_type,
            tanh_logit=cfg.tanh_logit,
            no_advantage=cfg.no_advantage,
            freeze_fe=cfg.freeze_fe,
            feature_extractor=feature_extractor,
        )
        q_net_target = copy.deepcopy(q_net)
        q_net.to(device)
        q_net_target.to(device)
        q_opt = torch.optim.Adam(q_net.parameters(), lr=cfg.q_lr)

        # A replay buffer stores experience collected over all sampling runs
        buffer = ReplayBuffer(
            torch.Size((int(act_space.n),)),
            cfg.buffer_size,
            cfg.last_step_sample_bonus > 1,
        )

        obs_, info = env.reset()
        obs = process_obs(obs_)
        mask = process_act_masks(obs_)
        warmup_steps = int(cfg.buffer_size / cfg.train_steps)
        traj_id = 0
        for step in tqdm(range(warmup_steps + cfg.iterations), position=0):
            train_step = max(step - warmup_steps, 0)
            percent_done = max((step - warmup_steps) / cfg.iterations, 0)
            last_step_sample_bonus = cfg.last_step_sample_bonus + (cfg.last_step_sample_bonus_start - cfg.last_step_sample_bonus) * max(0, 1 - (train_step / cfg.last_step_sample_bonus_start_steps))

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
                        obs.to(device)
                        action, _ = get_action(q_net, obs, mask.to(device))
                        obs.to("cpu")
                    obs_, reward, done, trunc, info_ = env.step(action)

                    # Normalize reward if last step
                    if (done or trunc) and not cfg.use_potential:
                        reward = (reward - cfg.norm_min) / (cfg.norm_max - cfg.norm_min)

                    next_obs = process_obs(obs_)
                    next_mask = process_act_masks(obs_)
                    inserted_idx = buffer.next
                    buffer.insert_step(
                        [obs.to_data_list()[0]],
                        [next_obs.to_data_list()[0]],
                        torch.tensor([action]),
                        [reward],
                        [done],
                        [
                            (
                                True
                                if not cfg.distr_scorer
                                else ((not cfg.use_potential) and (not (done or trunc)))
                            )
                        ],
                        [1.0 if done or trunc else last_step_sample_bonus],
                        [traj_id]
                    )
                    scorer.update(buffer)
                    
                    # Send model to be scored (may be a no-op)
                    if cfg.use_potential or (done or trunc):
                        scorer.push_model(env.model, inserted_idx, env.label_idx, traj_id)

                    obs = next_obs
                    mask = next_mask
                    if done or trunc:
                        obs_, info = env.reset()
                        obs = process_obs(obs_)
                        mask = process_act_masks(obs_)
                        if not cfg.distr_scorer and cfg.use_traj_returns:
                            buffer.set_traj_return(traj_id, reward)
                        traj_id = (traj_id + 1) % cfg.buffer_size

            # Train
            if buffer.filled:
                # Unfreeze feature extractor if unfreezing is specified
                if cfg.freeze_fe_for > 0 and train_step == cfg.freeze_fe_for:
                    q_net.freeze_fe(False)

                total_q_loss = train_dqn(
                    q_net,
                    q_net_target,
                    q_opt,
                    buffer,
                    device,
                    cfg.train_iters,
                    cfg.train_batch_size,
                    cfg.discount,
                    cfg.use_traj_returns,
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
                        images = {p: [] for p in prompts}
                        for eval_run_id in range(cfg.eval_steps):
                            steps_taken = 0
                            episode_reward = 0.0
                            for _ in range(cfg.max_eval_steps):
                                action, q_val = get_action(q_net, eval_obs.to(device), eval_mask.to(device))
                                pred_reward_total += q_val
                                obs_, reward, done, trunc, _ = test_env.step(action)
                                eval_obs = eval_obs = process_obs(obs_)
                                eval_mask = process_act_masks(obs_)
                                steps_taken += 1
                                reward_total += reward
                                episode_reward += reward
                                if done or trunc:
                                    images[test_env.prompt].append((episode_reward, test_env.screenshot()[0]))

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
                            "eval_images": sum([[wandb.Image(img, caption=f"{k}: {score}") for (score, img) in imgs] for k, imgs in images.items() if len(imgs) > 0], []),
                            "last_step_sample_bonus": last_step_sample_bonus,
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
