from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import *
import gymnasium as gym
from matplotlib import pyplot as plt
import rerun as rr
from safetensors.torch import load_model

import numpy as np
import torch
from kitbasher import train
from kitbasher.train import (
    LABELS,
    QNet,
    connect_scorer,
    connect_start,
    create_clip_scorer,
    get_action,
    process_act_masks,
    process_obs,
    single_start,
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
    max_actions_per_step: int = 100
    prompt: str = "a lego "
    checkpoint: str = ""
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
        start_fn = single_start
    elif cfg.score_fn == "connect":
        score_fn = connect_scorer
        start_fn = connect_start
    elif cfg.score_fn == "clip":
        score_fn = create_clip_scorer()
        start_fn = single_start
    else:
        raise NotImplementedError(f"Invalid score function, got {cfg.score_fn}")
    prompts = [cfg.prompt + l for l in LABELS]
    env = ConstructionEnv(
        score_fn=score_fn,
        start_fn=start_fn,
        max_actions_per_step=cfg.max_actions_per_step,
        use_potential=cfg.use_potential,
        max_steps=cfg.max_steps,
        visualize=True,
        prompts=prompts,
    )

    # Initialize Q network
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Graph)
    assert isinstance(obs_space.node_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Discrete)
    if cfg.checkpoint:
        with open(Path(cfg.checkpoint).parent / "meta.json", "r") as f:
            meta_json = json.load(f)
        train_cfg = train.Config.model_validate_json(meta_json)
        q_net = QNet(
            env.num_parts,
            32,
            train_cfg.process_layers,
            obs_space.node_space.shape[0],
            64,
            train_cfg.process_type,
            train_cfg.tanh_logit,
        )
        load_model(q_net, cfg.checkpoint)
    with torch.no_grad():
        obs_, info = env.reset()
        env.render()
        eval_obs = process_obs(obs_)
        eval_mask = process_act_masks(obs_)
        for _ in range(cfg.max_eval_steps):
            if cfg.checkpoint:
                action, q_val = get_action(q_net, eval_obs, eval_mask)
            else:
                action = action = random.choice(
                    [i for i, b in enumerate((~eval_mask.bool()).tolist()) if b]
                )
            obs_, reward, done, trunc, info = env.step(action)
            env.render()
            rr.log(
                "volume", rr.Boxes3D(half_sizes=[40.0, 10.0, 5.0], centers=[0, 0, 0])
            )

            # Show model scoring screenshot
            plt.imshow(env.screenshot()[0])
            plt.show()
            # plt.imshow(env.screenshot()[1])
            # plt.show()
            print(reward)

            eval_obs = eval_obs = process_obs(obs_)
            eval_mask = process_act_masks(obs_)
            if done or trunc:
                obs_, info = env.reset()
                eval_obs = process_obs(obs_)
                eval_mask = process_act_masks(obs_)
                break
