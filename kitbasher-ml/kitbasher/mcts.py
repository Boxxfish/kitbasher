import math
from typing import *
import torch
from torch import nn

from kitbasher.env import ConstructionEnv, EnvState
from kitbasher.utils import get_action

class MCTSNode:
    """
    Stores a Monte Carlo Tree Search node.
    """

    def __init__(self, value: float, num_actions: int, discount: float, c_puct: float):
        self.total_return = value
        self.visited = 0
        self.discount = discount
        self.c_puct = c_puct
        self.num_actions = num_actions
        self.children: Optional[list[MCTSNode]] = None
        self.rewards: Optional[List[Optional[float]]] = None
        self.done = False
        self.state = None
        self.action_offset = None # To account for masking

    def simulate(self, q_net: nn.Module, env: ConstructionEnv) -> Optional[Tuple[EnvState, float]]:
        """
        Runs simulate step on this node, expanding children as needed.
        """
        env.load_state(self.state)
        print(env.timer)
        
        if not self.children:
            # Expand child nodes when first expanding
            obs = env.gen_obs()
            mask = torch.tensor([1] * len(env.model) + [0] * len(env.place_configs))
            q_vals = q_net(obs).squeeze(1)
            self.children = []
            for i in range(len(q_vals)):
                if mask[i].item():
                    continue
                elif not self.action_offset:
                    self.action_offset = i
                self.children.append(MCTSNode(q_vals[i].item(), self.num_actions, self.discount, self.c_puct))
            self.rewards = [None] * len(self.children)
        
        # Choose action and simulate next step
        total_visited_sqrt = math.sqrt(sum([x.visited for x in self.children]))
        action = max(enumerate(self.children), key=lambda x: x[1].puct(total_visited_sqrt))[0]
        
        if self.rewards[action] is None:
            _, reward, done_, trunc_, _ = env.step(self.action_offset + action)
            self.rewards[action] = reward
            self.children[action].state = env.get_state()
            done = done_ or trunc_
        else:
            reward = self.rewards[action]
            done = self.children[action].done
        
        subsequent_return = 0.0
        done_results = None
        if done:
            self.children[action].done = True
            done_results = (self.children[action].state, self.rewards[action])
        elif self.children[action].visited == 0:
            # Don't recurse
            subsequent_return = self.children[action].avg_value()
            self.children[action].visited += 1
        else:
            # Expand child node if not done
            done_results = self.children[action].simulate(q_net, env)
            subsequent_return = self.children[action].avg_value()
            
        self.total_return += reward + self.discount * subsequent_return
        self.visited += 1
        return done_results

    def avg_value(self) -> float:
        """
        Returns the average value experienced by this node.
        """
        if self.visited == 0:
            return self.total_return
        return self.total_return / self.visited

    def puct(self, total_visited_sqrt: float) -> float:
        """
        Returns the PUCT score.
        Currently assumes uniform probablities.
        If this node is done, returns -1 to prevent re-expansion.
        """
        if self.done:
            return -1
        q = self.avg_value()
        u = self.c_puct * (1 / self.num_actions) * total_visited_sqrt / (1 + self.visited)
        return q + u

def run_mcts(
    q_net: nn.Module,
    env: ConstructionEnv,
    initial_state: EnvState,
    rollouts: int,
    discount: float,
    num_actions: int,
    c_puct: float = 4.0,
) -> Tuple[EnvState, float]:
    """
    Runs MCTS on the provided env, and returns the sequence with the highest reward.

    c_puct: Higher values encourage exploration.
    """
    # Root node is special case, we'll have the first expansion set its actual value
    root = MCTSNode(0.0, num_actions, discount, c_puct)
    root.state = initial_state
    root.visited = 0
    best_sol = None
    best_score = -float("inf")
    rollout_num = 0
    while best_sol is None or rollout_num < rollouts:
        print(f"Rollout {rollout_num}")
        result = root.simulate(q_net, env)

        # If a solution was returned, compare it against the previous solution
        if result:
            print("")
            print(result[1])
            print("")
            new_sol, new_score = result
            if new_score > best_score:
                best_sol = new_sol
                best_score = new_score
        
        rollout_num += 1
    print("Finished rollouts")
    return (best_sol, best_score)