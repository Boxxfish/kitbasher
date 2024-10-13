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
from safetensors.torch import save_model

from kitbasher.algorithms.dqn import train_dqn
from kitbasher.algorithms.replay_buffer import ReplayBuffer
from kitbasher.env import ConstructionEnv

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
    eval_every: int = 100
    out_dir: str = "runs"
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
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: Data):
        data = data.sort()
        x, edge_index, batch, part_ids = (
            data.x,
            data.edge_index,
            data.batch,
            data.part_ids,
        )
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
        advantage_mean = self.mean_aggr(advantage, batch)  # Shape: (num_batches, 1)
        advantage_mean = torch.gather(
            advantage_mean, 0, batch.unsqueeze(1)
        )  # Shape: (num_nodes, 1)
        value_x = self.mean_aggr(x, batch)  # Shape: (num_batches, hidden_dim)
        value = self.value(value_x)  # Shape: (num_batches, 1)
        value = torch.gather(value, 0, batch.unsqueeze(1))  # Shape: (num_nodes, 1)
        return value + advantage - advantage_mean


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


def single_start(engine: EngineWrapper):
    config = engine.create_config(5, 0, 0, 0)
    engine.place_part(config)


def volume_fill_scorer(
    model: List[PyPlacedConfig], data: Data, env: "ConstructionEnv", is_done: bool
) -> tuple[float, bool]:
    """
    Grants a reward of 1 for every part that touches the volume.
    """
    cx, cy, cz = [0.0, 0.0, 0.0]
    hx, hy, hz = [40.0, 10.0, 5.0]
    score = 0
    for placed in model:
        part_score = 1
        for bbox in placed.bboxes:
            if (
                abs(placed.position.x + bbox.center.x - cx) > (hx + bbox.half_sizes.x)
                or abs(placed.position.y + bbox.center.y - cy)
                > (hy + bbox.half_sizes.y)
                or abs(placed.position.z + bbox.center.z - cz)
                > (hz + bbox.half_sizes.z)
            ):
                part_score = 0
                break
        score += part_score
    return score, False


def connect_start(engine: EngineWrapper):
    # Generate a model with 20 pieces
    parts: List[PyPlacedConfig] = []
    config = engine.create_config(5, 0, 0, 0)
    engine.place_part(config)
    parts.append(config)
    for _ in range(20):
        candidates = engine.gen_candidates()
        config = random.choice(candidates)
        parts.append(config)
        engine.place_part(config)

    engine.clear_model()

    # Place two random parts of the model
    indices = list(range(0, len(parts)))
    random.shuffle(indices)
    for i in range(2):
        part = parts[indices[i]]
        part.connections = [None for _ in part.connections]
        engine.place_part(part)


def connect_scorer(
    model: List[PyPlacedConfig], data: Data, env: "ConstructionEnv", is_done: bool
) -> tuple[float, bool]:
    """
    Returns 1 and ends the episode if the model is connected.
    """
    # Set up adjacency list
    edges: Dict[int, List[int]] = {}
    for e1, e2 in data.edge_index.T.tolist():
        if e1 >= len(model) or e2 >= len(model):
            continue
        if e1 not in edges:
            edges[e1] = []
        if e2 not in edges:
            edges[e2] = []
        edges[e1].append(e2)
        edges[e2].append(e1)

    # Perform DFS to check number of nodes reachable from node 0
    seen = set()
    stack = [0]
    while len(stack) > 0:
        node_idx = stack.pop()
        if node_idx in edges:
            for neighbor in edges[node_idx]:
                if neighbor not in seen:
                    stack.append(neighbor)
            seen.add(node_idx)

    connected = len(seen) == len(model)

    return 1.0 if connected else 0.0, connected


def create_clip_scorer(model_url: str = "openai/clip-vit-base-patch32"):
    from transformers import CLIPProcessor, CLIPModel

    clip = CLIPModel.from_pretrained(model_url)
    processor = CLIPProcessor.from_pretrained(model_url)

    def clip_scorer(
        model: List[PyPlacedConfig], data: Data, env: "ConstructionEnv", is_done: bool
    ) -> tuple[float, bool]:
        """
        Returns the score returned by CLIP.
        """
        if not is_done:
            return 0.0, False
        prompt = env.prompt
        imgs = env.screenshot()
        inputs = processor(
            text=[prompt],
            images=imgs,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
        )

        outputs = clip(**inputs)
        logits_per_image = outputs.logits_per_image
        score = logits_per_image.mean().item() / 30.0
        return score, False

    return clip_scorer


if __name__ == "__main__":
    cfg = Config()
    parser = ArgumentParser()
    for k, v in cfg.model_fields.items():
        if v.annotation == bool:
            parser.add_argument(
                f"--{k.replace('_', '-')}", default=v.default, action="store_true"
            )
        else:
            parser.add_argument(
                f"--{k.replace('_', '-')}", default=v.default, type=v.annotation
            )
    args = parser.parse_args()
    cfg = Config(**args.__dict__)
    device = torch.device(cfg.device)

    wandb.init(
        project="kitbasher",
        config=cfg.__dict__,
    )

    # Create out directory
    assert wandb.run is not None
    for _ in range(100):
        if wandb.run.name != "":
            break
    if wandb.run.name != "":
        out_id = wandb.run.name
    else:
        out_id = "testing"

    out_dir = Path(cfg.out_dir)
    exp_dir = out_dir / out_id
    try:
        os.mkdir(exp_dir)
    except OSError as e:
        print(e)
    meta = ExpMeta(args=cfg)
    with open(exp_dir / "meta.json", "w") as f:
        json.dump(meta.model_dump_json(), f)

    chkpt_path = exp_dir / "checkpoints"
    try:
        os.mkdir(chkpt_path)
    except OSError as e:
        print(e)

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
    )
    test_env = ConstructionEnv(
        score_fn=score_fn,
        start_fn=start_fn,
        max_actions_per_step=cfg.max_actions_per_step,
        use_potential=cfg.use_potential,
        max_steps=cfg.max_steps,
        prompts=prompts,
    )

    # Initialize Q network
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Graph)
    assert isinstance(obs_space.node_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Discrete)
    q_net = QNet(
        env.num_parts,
        32,
        cfg.process_layers,
        obs_space.node_space.shape[0],
        64,
        cfg.process_type,
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
                    obs_, _ = test_env.reset()
                    eval_obs = process_obs(obs_)
                    eval_mask = process_act_masks(obs_)
                    for _ in range(cfg.eval_steps):
                        steps_taken = 0
                        for _ in range(cfg.max_eval_steps):
                            action, q_val = get_action(q_net, eval_obs, eval_mask)
                            pred_reward_total += q_val
                            obs_, reward, done, trunc, _ = test_env.step(action)
                            eval_obs = eval_obs = process_obs(obs_)
                            eval_mask = process_act_masks(obs_)
                            steps_taken += 1
                            reward_total += reward
                            if done or trunc:
                                obs_, info = test_env.reset()
                                eval_obs = process_obs(obs_)
                                eval_mask = process_act_masks(obs_)
                                break
                log_dict.update(
                    {
                        "avg_eval_episode_reward": reward_total / cfg.eval_steps,
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
