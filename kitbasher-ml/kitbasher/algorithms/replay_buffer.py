"""
A replay buffer for use with off policy algorithms.
"""
from typing import List, Tuple
from torch_geometric.data import Data # type: ignore

import torch


class ReplayBuffer:
    """
    Stores transitions and generates mini batches.
    """

    def __init__(
        self,
        action_masks_shape: torch.Size,
        capacity: int,
    ):
        k = torch.float
        action_shape = torch.Size([capacity])
        action_masks_shape = torch.Size([capacity] + list(action_masks_shape))
        self.capacity = capacity
        self.next = 0
        d = torch.device("cpu")
        self.states = [None] * self.capacity
        self.next_states = [None] * self.capacity
        self.actions = torch.zeros(
            action_shape, dtype=torch.int64, device=d, requires_grad=False
        )
        self.rewards = torch.zeros([capacity], dtype=k, device=d, requires_grad=False)
        self.scores = torch.zeros([capacity], dtype=k, device=d, requires_grad=False)
        # Technically this is the "terminated" flag
        self.dones = torch.zeros([capacity], dtype=k, device=d, requires_grad=False)
        self.readys = torch.zeros([capacity], dtype=torch.bool, device=d, requires_grad=False)
        self.filled = False

    def insert_step(
        self,
        states: List[Data],
        next_states: List[Data],
        actions: torch.Tensor,
        rewards: List[float],
        dones: List[bool],
        readys: List[bool],
    ):
        """
        Inserts a transition from each environment into the buffer. Make sure
        more data than steps aren't inserted.
        """
        batch_size = len(dones)
        d = torch.device("cpu")
        with torch.no_grad():
            indices = torch.arange(
                self.next,
                (self.next + batch_size),
            ).remainder(self.capacity)
            for j, i in enumerate(indices):
                idx = i.item()
                self.states[idx] = states[j]
                self.next_states[idx] = next_states[j]
            self.actions.index_copy_(0, indices, actions)
            self.rewards.index_copy_(
                0, indices, torch.tensor(rewards, dtype=torch.float, device=d)
            )
            self.scores.index_copy_(
                0, indices, torch.tensor([0.0] * batch_size, dtype=torch.float, device=d)
            )
            self.dones.index_copy_(
                0, indices, torch.tensor(dones, dtype=torch.float, device=d)
            )
            self.readys.index_copy_(
                0, indices, torch.tensor(readys, dtype=torch.bool, device=d)
            )
        self.next = (self.next + batch_size) % self.capacity
        if self.next == 0:
            self.filled = True

    def sample(
        self, batch_size: int
    ) -> Tuple[
        List[Data],
        List[Data],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generates minibatches of experience.
        """
        with torch.no_grad():
            indices = torch.arange(0, self.capacity)
            indices = indices[self.readys]
            indices = indices[torch.randperm(indices.shape[0])][:batch_size]
            rand_states = []
            rand_next_states = []
            for i in indices:
                idx = i.item()
                rand_states.append(self.states[idx])
                rand_next_states.append(self.next_states[idx])
            rand_actions = self.actions.index_select(0, indices)
            rand_rewards = self.rewards.index_select(0, indices)
            rand_dones = self.dones.index_select(0, indices)
            return (
                rand_states,
                rand_next_states,
                rand_actions,
                rand_rewards,
                rand_dones,
            )
