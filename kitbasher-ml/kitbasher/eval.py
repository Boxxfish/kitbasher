from argparse import ArgumentParser
from dataclasses import dataclass
from typing import *
import gymnasium as gym

import torch
from kitbasher.train import (
    QNet,
    get_action,
    process_act_masks,
    process_obs,
    volume_fill_scorer,
)
from kitbasher.env import ConstructionEnv


@dataclass
class Config:
    max_steps: int = 8
    score_fn: str = "volume"  # Score function to use. Choices: "volume", "clip".
    use_potential: bool = (
        False  # If true, the agent is rewarded by the change in scoring on each timestep. Otherwise, the reward is only given at the end.
    )
    max_eval_steps: int = 8
    device: str = "cuda"


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

    if cfg.score_fn == "volume":
        score_fn = volume_fill_scorer
    else:
        raise NotImplementedError(f"Invalid score function, got {cfg.score_fn}")
    env = ConstructionEnv(
        score_fn=score_fn, use_potential=cfg.use_potential, max_steps=cfg.max_steps
    )

    # Initialize Q network
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Graph)
    assert isinstance(obs_space.node_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Discrete)
    q_net = QNet(1, obs_space.node_space.shape[0], 4)
    with torch.no_grad():
        obs_, info = env.reset()
        eval_obs = process_obs(obs_)
        eval_mask = process_act_masks(obs_)
        for _ in range(cfg.max_eval_steps):
            action, q_val = get_action(q_net, eval_obs, eval_mask)
            obs_, reward, done, trunc, info = env.step(action)
            eval_obs = eval_obs = process_obs(obs_)
            eval_mask = process_act_masks(obs_)
            if done or trunc:
                obs_, info = env.reset()
                eval_obs = process_obs(obs_)
                eval_mask = process_act_masks(obs_)
                break
