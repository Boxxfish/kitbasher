import time
from typing import *
from kitbasher_rust import PyPlacedConfig
import torch
import zmq
import subprocess

import gymnasium as gym

from kitbasher.algorithms.replay_buffer import ReplayBuffer
from kitbasher.batch_scoring.messages import RenderMessage, ScoredMessage
from kitbasher.env import BLOCK_PARTS, ConstructionEnv
from kitbasher.scorers import connect_scorer, connect_start, create_clip_scorer, create_contrastive_clip_scorer, dummy_scorer, single_start, volume_fill_scorer

class DistributedScorer:
    """
    Allows for distributed scoring of models. Once a model is scored, it will be rendered and scored in a different
    process, and the score in the experience buffer will be updated.

    If an exception is raised and this is used as a context manager, workers will automatically be killed.
    """

    def __init__(
        self,
        max_queued_items: int,
        part_paths: list[str],
        use_mirror: bool,
        prompts: list[str],
        scorer_fn: str,
        norm_min: float,
        norm_max: float,
        num_render_workers: int = 2,
        train_port_out: int = 5557,
        score_port_in: int = 5558,
        train_port_in: int = 5559,
    ) -> None:
        self.max_queued_items = max_queued_items
        self.num_queued_items = 0
        self.prompts = prompts
        self.scorer_fn = scorer_fn
        self.norm_max = norm_max
        self.norm_min = norm_min

        # Set up sockets
        context = zmq.Context()

        self.sender = context.socket(zmq.PUSH)
        self.sender.bind(f"tcp://*:{train_port_out}")

        self.receiver = context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://*:{train_port_in}")
        self.receiver.RCVTIMEO = 0 # If a message isn't present within this time, break 

        # Launch render workers and scorer
        self.scorer_proc = subprocess.Popen(
            [
                "python",
                "-m",
                "kitbasher.batch_scoring.scorer",
                "--score-port-in",
                str(score_port_in),
                "--train-port-in",
                str(train_port_in),
            ]
        )
        self.render_procs = []
        for _ in range(num_render_workers):
            args = ["cargo", "run", "--manifest-path", "kitbasher_rust/Cargo.toml", "--bin", "render_worker", "--release", "--"]
            for part_path in part_paths:
                args.extend(["-p", part_path.replace(".ron", ".glb")])
            if use_mirror:
                args.append("--use-mirror")
            args.extend(["--in-socket-addr", str(train_port_out)])
            args.extend(["--out-socket-addr", str(score_port_in)])
            render_proc = subprocess.Popen(args)
            self.render_procs.append(render_proc)

    def push_model(self, model: list[PyPlacedConfig], buffer_idx: int, label_idx: int):
        """Sends a model to be rendered and scored."""
        self.sender.send_json(
            RenderMessage(
                buffer_idx=buffer_idx,
                label_idx=label_idx,
                part_configs=[part_config.to_json() for part_config in model],
                prompts=self.prompts,
                scorer_fn=self.scorer_fn,
            ).model_dump()
        )
        self.num_queued_items += 1

    def update(self, buffer: ReplayBuffer):
        """Checks for items in the queue and updates the buffer."""
        need_fill = self.num_queued_items == self.max_queued_items
        while True:
            try:
                scored_msg = ScoredMessage.model_validate(self.receiver.recv_json())
                buffer.readys[scored_msg.buffer_idx] = True
                buffer.rewards[scored_msg.buffer_idx] = (scored_msg.score - self.norm_min) / (self.norm_max - self.norm_min)
                self.num_queued_items -= 1
            except zmq.Again:
                # If we've hit the max for queued items, keep polling until we catch up.
                # Otherwise, exit the loop to prevent blocking.
                if need_fill and self.num_queued_items > 0:
                    pass
                else:
                    break

    def destroy(self):
        self.scorer_proc.kill()
        for render_proc in self.render_procs:
            render_proc.kill()

    def __enter__(self) -> "DistributedScorer":
        return self

    def __exit__(self, type, value, traceback):
        try:
            pass
        except Exception as e:
            raise e
        finally:
            self.destroy()

class DummyDistributedScorer:
    """No-op version of `DistributedScorer`."""

    def push_model(self, model: list[PyPlacedConfig], buffer_idx: int, label_idx: int):
        pass

    def update(self, buffer: ReplayBuffer):
        pass

    def destroy(self):
        pass

    def __enter__(self) -> "DummyDistributedScorer":
        return self
    
    def __exit__(self, type, value, traceback):
        pass

def get_scorer_fn(
    score_fn_name: str,
    distr_scorer: bool,
    max_queued_items: int,
    use_mirror: bool,
    prompts: list[str],
    num_render_workers: int,
    part_paths: list[str],
    norm_min: float,
    norm_max: float,
) -> Tuple[Any, Any, Any, Callable[[], DistributedScorer]]:
    if score_fn_name == "volume":
        score_fn = volume_fill_scorer
        eval_score_fn = score_fn
        start_fn = single_start
        scorer = lambda: DummyDistributedScorer()
    elif score_fn_name == "connect":
        score_fn = connect_scorer
        eval_score_fn = score_fn
        start_fn = connect_start
        scorer = lambda: DummyDistributedScorer()
    elif score_fn_name == "clip":
        if distr_scorer:
            score_fn = dummy_scorer
            eval_score_fn = create_clip_scorer()
            start_fn = single_start
            scorer = lambda: DistributedScorer(
                max_queued_items=max_queued_items,
                use_mirror=use_mirror,
                prompts=prompts,
                num_render_workers=num_render_workers,
                scorer_fn=score_fn_name,
                part_paths=part_paths,
                norm_min=norm_max,
                norm_min=norm_min,
            )
        else:
            score_fn = create_clip_scorer()
            eval_score_fn = score_fn
            start_fn = single_start
            scorer = lambda: DummyDistributedScorer()
    elif score_fn_name == "contrastive_clip":
        if distr_scorer:
            score_fn = dummy_scorer
            eval_score_fn = create_contrastive_clip_scorer()
            start_fn = single_start
            scorer = lambda: DistributedScorer(
                max_queued_items=max_queued_items,
                use_mirror=use_mirror,
                prompts=prompts,
                num_render_workers=num_render_workers,
                scorer_fn=score_fn_name,
                part_paths=part_paths,
                norm_min=norm_max,
                norm_min=norm_min,
            )
        else:
            score_fn = create_contrastive_clip_scorer()
            eval_score_fn = score_fn
            start_fn = single_start
            scorer = lambda: DummyDistributedScorer()
    else:
        raise NotImplementedError(f"Invalid score function, got {score_fn_name}")
    return score_fn, eval_score_fn, start_fn, scorer

# Small test
if __name__ == "__main__":
    use_mirror = True
    steps = 16
    prompts = ["a lego car", "a lego chair"]
    env = ConstructionEnv(volume_fill_scorer, single_start, False, 1, use_mirror, steps, False, prompts)
    
    # Set up buffer
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete)
    buffer = ReplayBuffer(torch.Size((int(act_space.n),)), 10)
    
    # Sample and score
    env.reset()
    for _ in range(steps):
        env.step(len(env.model))
    with DistributedScorer(4, BLOCK_PARTS, use_mirror, prompts, "clip") as scorer:
        for i in range(8):
            scorer.push_model(env.model, i, 0)
            time.sleep(0.01) # Simulate delay
            scorer.update(buffer)
        print("Rewards in buffer:", buffer.rewards.tolist())
