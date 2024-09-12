from typing import *
import gymnasium as gym
import gymnasium as gym
import torch
from torch import Tensor
from kitbasher_rust import EngineWrapper, PyAABB, PyPlacedConfig
from torch_geometric.data import Data  # type: ignore

BLOCK_PARTS = [
    "../kitbasher-game/assets/models/1x1.ron",
    "../kitbasher-game/assets/models/2x1_slanted.ron",
    "../kitbasher-game/assets/models/2x1.ron",
    "../kitbasher-game/assets/models/2x2_axle.ron",
    "../kitbasher-game/assets/models/2x2_slanted.ron",
    "../kitbasher-game/assets/models/2x2.ron",
    "../kitbasher-game/assets/models/4x1.ron",
    "../kitbasher-game/assets/models/wheel.ron",
]
BLOCK_CONNECT_RULES = [(0, 0), (1, 2)]

MAX_CONNECTIONS = 12
CONNECTION_DIM = 3 + 3 + 3 + 1 + 1
NODE_DIM = 6 + 4 + 1 + MAX_CONNECTIONS * CONNECTION_DIM


class ConstructionEnv(gym.Env):
    def __init__(self, max_steps: Optional[int] = None) -> None:
        self.engine = EngineWrapper(BLOCK_PARTS, BLOCK_CONNECT_RULES)
        self.model: List[PyPlacedConfig] = []
        self.place_configs: List[PyPlacedConfig] = []
        self.timer = 0
        self.max_steps = max_steps

    def step(self, action: int) -> tuple[Data, float, bool, bool, dict[str, Any]]:
        config = self.place_configs[action - len(self.model)]
        self.engine.place_part(config)
        self.timer += 1
        done = self.timer == self.max_steps
        obs, mask = self.gen_obs()
        return obs, 0.0, done, False, {"action_mask": mask}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Data, dict[str, Any]]:
        self.engine.clear_model()
        config = self.engine.create_config(5, 0, 0, 0)
        self.engine.place_part(config)
        self.timer = 0
        obs, mask = self.gen_obs()
        return obs, {"action_mask": mask}

    def gen_obs(self) -> Tuple[Data, Tensor]:
        self.model = self.engine.get_model()
        self.place_configs = self.engine.gen_candidates()
        part_ids = []
        nodes = []
        edges = set()
        for j, config in enumerate(self.model + self.place_configs):
            part_ids.append(config.part_id)
            min_bbox, max_bbox = merge_bboxes(config.bboxes)
            node_vec = torch.zeros([NODE_DIM])
            node_vec[0] = min_bbox[0]
            node_vec[1] = min_bbox[1]
            node_vec[2] = min_bbox[2]
            node_vec[3 + 0] = max_bbox[0]
            node_vec[3 + 1] = max_bbox[1]
            node_vec[3 + 2] = max_bbox[2]
            rot_start = 6
            node_vec[rot_start + 0] = config.rotation.x
            node_vec[rot_start + 1] = config.rotation.y
            node_vec[rot_start + 2] = config.rotation.z
            node_vec[rot_start + 3] = config.rotation.w
            node_vec[rot_start + 4] = int(j >= len(self.place_configs))
            conn_start_all = rot_start + 4 + 1
            for i in range(MAX_CONNECTIONS):
                if i < len(config.connectors):
                    conn_start = conn_start_all + i * CONNECTION_DIM
                    connection = config.connections[i]
                    connector = config.connectors[i]
                    node_vec[conn_start + connector.connect_type] = 1
                    node_vec[conn_start + 3 + 0] = connector.position.x
                    node_vec[conn_start + 3 + 1] = connector.position.y
                    node_vec[conn_start + 3 + 2] = connector.position.z
                    node_vec[conn_start + 6 + int(connector.axis)] = 1
                    node_vec[conn_start + 9] = int(connector.side_a)
                    node_vec[conn_start + 10] = int(connection is not None)
                    if connection is not None:
                        edges.add((j, connection.placed_id))
                        edges.add((connection.placed_id, j))
            nodes.append(node_vec)
        x = torch.stack(nodes)
        edge_index = torch.tensor([list(e) for e in edges]).T.contiguous()
        data = Data(x=x, edge_index=edge_index, part_ids=part_ids)
        mask = torch.tensor([1] * len(self.model) + [0] * len(self.place_configs))
        return data, mask


def merge_bboxes(bboxes: List[PyAABB]) -> Tuple[List[float], List[float]]:
    min_bbox = [float("inf")] * 3
    max_bbox = [-float("inf")] * 3
    for bbox in bboxes:
        center = [bbox.center.x, bbox.center.y, bbox.center.z]
        half_sizes = [bbox.half_sizes.x, bbox.half_sizes.y, bbox.half_sizes.z]
        min_ = [c - h for c, h in zip(center, half_sizes)]
        max_ = [c + h for c, h in zip(center, half_sizes)]
        min_bbox = [min(new, old) for new, old in zip(min_, min_bbox)]
        max_bbox = [max(new, old) for new, old in zip(max_, max_bbox)]
    return min_bbox, max_bbox
