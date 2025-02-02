"""
A replay buffer for use with off policy algorithms.
"""

from dataclasses import dataclass
import random
from typing import List, Tuple
from torch_geometric.data import Data  # type: ignore

import torch


@dataclass
class TrajData:
    expire_in: int
    traj_return: float


class ReplayBuffer:
    """
    Stores transitions and generates mini batches.
    """

    def __init__(
        self,
        action_masks_shape: torch.Size,
        capacity: int,
        use_weights: bool,  # Whether weights will be taken into account during sampling
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
        self.readys = torch.zeros(
            [capacity], dtype=torch.bool, device=d, requires_grad=False
        )
        self.filled = False
        self.use_weights = use_weights
        self.weights = torch.zeros(
            [capacity], dtype=torch.float, device=d, requires_grad=False
        )
        self.traj_ids = torch.zeros(
            [capacity], dtype=torch.int, device=d, requires_grad=False
        )
        self.traj_returns: dict[int, TrajData] = {}

    def insert_step(
        self,
        states: List[Data],
        next_states: List[Data],
        actions: torch.Tensor,
        rewards: List[float],
        dones: List[bool],
        readys: List[bool],
        weights: List[float],
        traj_ids: List[int],
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
                0,
                indices,
                torch.tensor([0.0] * batch_size, dtype=torch.float, device=d),
            )
            self.dones.index_copy_(
                0, indices, torch.tensor(dones, dtype=torch.float, device=d)
            )
            self.readys.index_copy_(
                0, indices, torch.tensor(readys, dtype=torch.bool, device=d)
            )
            self.weights.index_copy_(
                0, indices, torch.tensor(weights, dtype=torch.float, device=d)
            )
            self.traj_ids.index_copy_(
                0, indices, torch.tensor(traj_ids, dtype=torch.int, device=d)
            )
        self.next = (self.next + batch_size) % self.capacity
        if self.next == 0:
            self.filled = True

        for traj_id in list(self.traj_returns):
            self.traj_returns[traj_id].expire_in -= batch_size
            if self.traj_returns[traj_id].expire_in <= 0:
                self.traj_returns.pop(traj_id)

    def set_traj_return(self, traj_id: int, traj_return: float):
        assert traj_id not in self.traj_returns
        self.traj_returns[traj_id] = TrajData(self.capacity, traj_return)

    def sample(self, batch_size: int) -> Tuple[
        List[Data],
        List[Data],
        torch.Tensor,
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
            if not self.use_weights:
                indices = indices[torch.randperm(indices.shape[0])][:batch_size]
            else:
                weights = self.weights[indices].tolist()
                indices = torch.tensor(
                    random.choices(indices.tolist(), weights, k=batch_size)
                )
            rand_states = []
            rand_next_states = []
            for i in indices:
                idx = i.item()
                rand_states.append(self.states[idx])
                rand_next_states.append(self.next_states[idx])
            rand_actions = self.actions.index_select(0, indices)
            rand_rewards = self.rewards.index_select(0, indices)
            rand_dones = self.dones.index_select(0, indices)
            rand_traj_ids = self.traj_ids.index_select(0, indices)
            rand_traj_returns = torch.tensor(
                [
                    self.traj_returns[traj_id].traj_return if traj_id in self.traj_returns else 0.0
                    for traj_id in rand_traj_ids.tolist()
                ]
            )
            return (
                rand_states,
                rand_next_states,
                rand_actions,
                rand_rewards,
                rand_dones,
                rand_traj_returns,
            )
