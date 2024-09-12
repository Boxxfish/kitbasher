"""
Experiment for checking that DQN works.

The Deep Q Network (DQN) algorithm is a popular offline deep reinforcement
learning algorithm. It's intuitive to understand, and it gets reliable results,
though it can take longer to run.
"""

from argparse import ArgumentParser
import copy
import os
from pathlib import Path
import random
from functools import reduce
from typing import Any
import gymnasium as gym

import torch
import torch.nn as nn
import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from tqdm import tqdm
from safetensors.torch import save_model

from kitbasher.algorithms.dqn import train_dqn
from kitbasher.algorithms.replay_buffer import ReplayBuffer

_: Any
INF = 10**8


# Hyperparameters
class Config:
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
    warmup_steps: int = (
        500  # For the first n number of steps, we will only sample randomly.
    )
    buffer_size: int = 10_000  # Number of elements that can be stored in the buffer.
    target_update: int = 500  # Number of iterations before updating Q target.
    save_every: int = 100
    out_dir: str = "runs"
    device: str = "cuda"


class QNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        action_count: int,
    ):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.net = nn.Sequential(
            nn.Linear(flat_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.advantage = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_count)
        )
        self.value = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.action_count = action_count

    def forward(self, input: torch.Tensor):
        x = self.net(input)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


def get_action(q_net: nn.Module, obs: Any) -> tuple[int, float]:
    q_vals = q_net(obs).squeeze(0)
    action = q_vals.argmax(0).item()
    q_val = q_vals.amax(0).item()
    return action, q_val


def process_obs(obs: Any) -> Any:
    return torch.from_numpy(obs).unsqueeze(0)


def process_act_masks(info: Any) -> torch.Tensor:
    return torch.tensor([[0, 0]], dtype=torch.bool)#info["action_mask"]


if __name__ == "__main__":
    cfg = Config()
    parser = ArgumentParser()
    for k, v in cfg.__dict__.items():
        if isinstance(v, bool):
            parser.add_argument(
                f"--{k.replace('_', '-')}", default=v, action="store_true"
            )
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
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
    try:
        os.mkdir(out_dir / out_id)
    except OSError as e:
        print(e)
    chkpt_path = out_dir / out_id / "checkpoints"
    try:
        os.mkdir(chkpt_path)
    except OSError as e:
        print(e)

    env = CartPoleEnv()
    test_env = CartPoleEnv()

    # Initialize Q network
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete)
    q_net = QNet(obs_space.shape, int(act_space.n))
    q_net_target = copy.deepcopy(q_net)
    q_net_target.to(device)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=cfg.q_lr)

    # A replay buffer stores experience collected over all sampling runs
    buffer = ReplayBuffer(
        torch.Size(obs_space.shape),
        torch.Size((int(act_space.n),)),
        cfg.buffer_size,
    )

    obs_, info = env.reset()
    obs = process_obs(obs_)
    mask = process_act_masks(info)
    for step in tqdm(range(cfg.iterations), position=0):
        percent_done = step / cfg.iterations

        # Collect experience
        with torch.no_grad():
            for _ in range(cfg.train_steps):
                if (
                    random.random() < cfg.q_epsilon * max(1.0 - percent_done, 0.05)
                    or step < cfg.warmup_steps
                ):
                    action = act_space.sample()
                else:
                    action, _ = get_action(q_net, obs)
                obs_, reward, done, trunc, info_ = env.step(action)
                next_obs = process_obs(obs_)
                next_mask = process_act_masks(info_)
                buffer.insert_step(
                    obs,
                    next_obs,
                    torch.tensor([action]),
                    [reward],
                    [done],
                    mask,
                    next_mask,
                )
                obs = next_obs
                mask = next_mask

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

            # Evaluate the network's performance after this training iteration.
            with torch.no_grad():
                reward_total = 0
                pred_reward_total = 0
                obs_, info = test_env.reset()
                eval_obs = process_obs(obs_)
                eval_mask = process_act_masks(info)
                for _ in range(cfg.eval_steps):
                    steps_taken = 0
                    for _ in range(cfg.max_eval_steps):
                        action, q_val = get_action(q_net, eval_obs)
                        pred_reward_total += q_val
                        obs_, reward, done, trunc, info = test_env.step(action)
                        eval_obs = eval_obs = process_obs(obs_)
                        steps_taken += 1
                        reward_total += reward
                        if done or trunc:
                            obs_, info = test_env.reset()
                            eval_obs = process_obs(obs_)
                            eval_mask = process_act_masks(info)
                            break

            wandb.log(
                {
                    "avg_eval_episode_reward": reward_total / cfg.eval_steps,
                    "avg_eval_episode_predicted_reward": pred_reward_total
                    / cfg.eval_steps,
                    "avg_q_loss": total_q_loss / cfg.train_iters,
                    "q_lr": q_opt.param_groups[-1]["lr"],
                }
            )

            # Update Q target
            if step % cfg.target_update == 0:
                q_net_target.load_state_dict(q_net.state_dict())

            # Save checkpoint
            if step % cfg.save_every == 0:
                save_model(
                    q_net,
                    str(chkpt_path / f"q_net-{step}.safetensors"),
                )
