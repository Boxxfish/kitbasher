from typing import Tuple

import torch
from torch import nn
from torch_geometric import utils  # type: ignore
from torch_geometric.data import Batch  # type: ignore

from .replay_buffer import ReplayBuffer


def train_dqn(
    q_net: nn.Module,
    q_net_target: nn.Module,
    q_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
) -> float:
    """
    Performs the DQN training loop.
    Returns the total Q loss.
    """
    total_q_loss = 0.0
    q_net.train()
    if device.type != "cpu":
        q_net.to(device)

    total_q_loss = 0.0
    for _ in range(train_iters):
        (
            prev_states_,
            states_,
            actions,
            rewards,
            dones,
        ) = buffer.sample(train_batch_size)

        # Move batch to device if applicable
        prev_states = Batch.from_data_list(prev_states_).to(device=device)
        states = Batch.from_data_list(states_).to(device=device)
        actions = actions.to(device=device)
        rewards = rewards.to(device=device)
        dones = dones.to(device=device)

        # Train q network
        q_opt.zero_grad()
        prev_states_offsets_ = [0] + [
            s.x.shape[0] + 1 for s in prev_states.to_data_list()
        ][:-1]
        prev_states_offsets = torch.tensor(prev_states_offsets_, device=device).cumsum(
            0
        )  # Shape: (batch_size)
        with torch.no_grad():
            # Compute next actions
            q_vals = q_net(prev_states)  # Shape: (num_nodes, 1)
            q_vals = torch.masked_fill(
                q_vals, prev_states.action_mask.bool().unsqueeze(1), -torch.inf
            )  # Shape: (num_nodes, 1)
            unbatched_q_vals = utils.unbatch(q_vals, prev_states.batch, dim=0)
            next_actions_ = []
            offset = 0
            for q_vals in unbatched_q_vals:
                action = q_vals.argmax(0).item()
                next_actions_.append(offset + action)
                offset += q_vals.shape[0]
            next_actions = torch.tensor(
                next_actions_, device=device
            )  # Shape: (batch_size)
            q_target = rewards.unsqueeze(1) + discount * q_net_target(
                states
            ).detach().gather(0, next_actions.unsqueeze(1)) * (1.0 - dones.unsqueeze(1))
        diff = (
            q_net(prev_states).gather(0, (prev_states_offsets + actions).unsqueeze(1))
            - q_target
        )
        q_loss = (diff * diff).mean()
        q_loss.backward()
        q_opt.step()
        total_q_loss += q_loss.item()

    if device.type != "cpu":
        q_net.cpu()
    q_net.eval()
    return total_q_loss
