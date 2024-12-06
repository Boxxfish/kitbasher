from kitbasher_rust import PyPlacedConfig
import torch
import zmq
import subprocess

import gymnasium as gym

from kitbasher.algorithms.replay_buffer import ReplayBuffer
from kitbasher.batch_scoring.messages import RenderMessage, ScoredMessage
from kitbasher.env import BLOCK_PARTS, ConstructionEnv
from kitbasher.scorers import single_start, volume_fill_scorer


class DistributedScorer:
    """
    Allows for distributed scoring of models. Once a model is scored, it will be rendered and scored in a different
    process, and the score in the experience buffer will be updated.
    """

    def __init__(
        self,
        max_queued_items: int,
        part_paths: list[str],
        use_mirror: bool,
        prompts: list[str],
        num_render_workers: int = 2,
        train_port_out: int = 5557,
        score_port_in: int = 5558,
        train_port_in: int = 5559,
    ) -> None:
        self.max_queued_items = max_queued_items
        self.prompts = prompts

        # Set up sockets
        context = zmq.Context()

        self.sender = context.socket(zmq.PUSH)
        self.sender.bind(f"tcp://*:{train_port_out}")

        self.receiver = context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://*:{train_port_in}")

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

    def push_model(self, model: list[PyPlacedConfig], buffer_idx: int):
        """Sends a model to be rendered and scored."""
        self.sender.send_json(
            RenderMessage(
                buffer_idx=buffer_idx,
                label_idx=1,
                part_configs=[part_config.to_json() for part_config in model],
                prompts=self.prompts,
                scorer_fn="contrastive_clip",
            ).model_dump()
        )

    def update(self, buffer: ReplayBuffer):
        """Checks for items in the queue and updates the buffer."""
        scored_msg = ScoredMessage.model_validate(self.receiver.recv_json())
        print("Score:", scored_msg.score)

    def destroy(self):
        self.scorer_proc.kill()
        for render_proc in self.render_procs:
            render_proc.kill()


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
    scorer = DistributedScorer(8, BLOCK_PARTS, use_mirror, prompts)
    scorer.push_model(env.model, 0)
    scorer.update(buffer)
    scorer.destroy()
