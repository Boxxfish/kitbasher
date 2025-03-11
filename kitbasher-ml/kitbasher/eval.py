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
from kitbasher.mcts import run_mcts
from kitbasher.pretraining import Pretrained
from kitbasher.scorers import create_contrastive_clip_scorer, connect_scorer, connect_start, create_clip_scorer, single_start, volume_fill_scorer
from kitbasher.train import (
    LABELS,
    QNet,
    get_action,
    process_act_masks,
    process_obs,
)
from kitbasher.pretraining import ExpMeta as PretrainingExpMeta
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
    use_mirror: bool = False
    single_class: str = ""
    use_mcts: bool = False
    mcts_num_rollouts: int = 100
    mcts_c_puct: float = 4.0
    rerun: bool = False
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
    elif cfg.score_fn == "contrastive_clip":
        score_fn = create_contrastive_clip_scorer()
        start_fn = single_start
    else:
        raise NotImplementedError(f"Invalid score function, got {cfg.score_fn}")
    labels = LABELS
    add_steps = False
    if cfg.checkpoint:
        with open(Path(cfg.checkpoint).parent.parent / "meta.json", "r") as f:
            meta_json = f.read()
        train_cfg = train.ExpMeta.model_validate_json(meta_json).args
        add_steps =  train_cfg.add_steps
        if train_cfg.single_class:
            labels = [c.strip() for c in train_cfg.single_class.split(",")]
    prompts = [cfg.prompt + l for l in labels]

    env = ConstructionEnv(
        score_fn=score_fn,
        start_fn=start_fn,
        max_actions_per_step=cfg.max_actions_per_step,
        use_potential=cfg.use_potential,
        max_steps=cfg.max_steps,
        visualize=cfg.rerun,
        prompts=prompts,
        use_mirror=cfg.use_mirror,
        add_steps=add_steps,
    )

    # Initialize Q network
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Graph)
    assert isinstance(obs_space.node_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Discrete)
    if cfg.checkpoint:
        with open(Path(cfg.checkpoint).parent.parent / "meta.json", "r") as f:
            meta_json = f.read()
        train_cfg = train.ExpMeta.model_validate_json(meta_json).args
        feature_extractor = None
        if train_cfg.fe_path:
            meta_path = Path(train_cfg.fe_path).parent.parent / "meta.json"
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
            # Load model should update FE weights as well
            feature_extractor = pretrained.feature_extractor
        q_net = QNet(
            env.num_parts,
            32,
            train_cfg.process_layers,
            obs_space.node_space.shape[0],
            train_cfg.hidden_dim,
            train_cfg.process_type,
            train_cfg.tanh_logit,
            train_cfg.no_advantage,
            freeze_fe=train_cfg.freeze_fe,
            use_global_gcn=train_cfg.use_global_gcn,
            feature_extractor=feature_extractor,
        )
        load_model(q_net, cfg.checkpoint)
    with torch.no_grad():
        obs_, info = env.reset()
        env.render()
        print("Label:", env.prompts[env.label_idx])
        eval_obs = process_obs(obs_)
        eval_mask = process_act_masks(obs_)
        if not cfg.use_mcts:
            for _ in range(cfg.max_eval_steps):
                if cfg.checkpoint:
                    action, q_val = get_action(q_net, eval_obs, eval_mask)
                else:
                    action_choices = [i for i, b in enumerate((~eval_mask.bool()).tolist()) if b]
                    action = action = random.choice(action_choices)
                obs_, reward, done, trunc, info = env.step(action)
                env.render()
                rr.log(
                    "volume",
                    rr.Boxes3D(half_sizes=[40.0, 10.0, 5.0], centers=[0, 0, 0]),
                )

                # Show model scoring screenshot
                if done or trunc:
                    screenshots = env.screenshot()
                    plt.imshow(screenshots[0])
                    plt.show()
                    plt.imshow(screenshots[1])
                    plt.show()
                print(reward)

                eval_obs = eval_obs = process_obs(obs_)
                eval_mask = process_act_masks(obs_)
                if done or trunc:
                    obs_, info = env.reset()
                    eval_obs = process_obs(obs_)
                    eval_mask = process_act_masks(obs_)
                    break
        else:
            env.label_idx = 0
            best_sol, best_score = run_mcts(
                q_net,
                env,
                env.get_state(),
                cfg.mcts_num_rollouts,
                1.0,
                cfg.max_actions_per_step,
                cfg.mcts_c_puct,
            )
            print(best_score)
            env.load_state(best_sol)
            screenshots = env.screenshot()
            plt.imshow(screenshots[0])
            plt.show()
