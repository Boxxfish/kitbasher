from typing import *
import gymnasium as gym
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from kitbasher_rust import EngineWrapper, PyAABB, PyPlacedConfig
from torch_geometric.data import Data  # type: ignore
import rerun as rr  # type: ignore
import open3d as o3d  # type: ignore

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
MAX_NODES = 10_000


class PartModel:
    def __init__(self, path: str, part_index: int) -> None:
        model = o3d.io.read_triangle_model(path)
        self.vertices = np.asarray(model.meshes[0].mesh.vertices)
        self.normals = np.asarray(model.meshes[0].mesh.vertex_normals)
        self.triangles = np.asarray(model.meshes[0].mesh.triangles)
        self.part_index = part_index
        rr.log(
            f"model/parts/{self.part_index}",
            rr.Mesh3D(
                vertex_positions=self.vertices.tolist(),
                vertex_normals=self.normals.tolist(),
                triangle_indices=self.triangles.tolist(),
            ),
            rr.InstancePoses3D(
                scales=[0.0, 0.0, 0.0],  # Hide models by default
            ),
        )

    def render(self, translations: List[List[float]], rotations: List[List[float]]):
        rr.log(
            f"model/parts/{self.part_index}",
            rr.InstancePoses3D(
                translations=translations,
                quaternions=rotations,
                scales=[1.0, 1.0, 1.0] if len(translations) > 0 else [0.0, 0.0, 0.0],
            ),
        )


class ConstructionEnv(gym.Env):
    def __init__(
        self,
        score_fn: Callable[[List[PyPlacedConfig], Data], tuple[float, bool]],
        start_fn: Callable[[EngineWrapper], None],
        use_potential: bool,
        max_steps: Optional[int] = None,
        visualize: bool = False,
    ) -> None:
        self.num_parts = len(BLOCK_PARTS)
        self.engine = EngineWrapper(BLOCK_PARTS, BLOCK_CONNECT_RULES)
        self.model: List[PyPlacedConfig] = []
        self.place_configs: List[PyPlacedConfig] = []
        self.timer = 0
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Graph(
            node_space=gym.spaces.Box(-1, 1, [NODE_DIM]), edge_space=None
        )
        self.action_space = gym.spaces.Discrete(MAX_NODES)
        self.score_fn = score_fn
        self.start_fn = start_fn
        self.use_potential = use_potential
        self.last_score = 0.0
        if visualize:
            rr.init("Construction")
            rr.spawn()
            self.models: List[PartModel] = []
            for i, path in enumerate(BLOCK_PARTS):
                model = PartModel(path.replace(".ron", ".glb"), i)
                self.models.append(model)
            rr.log(
                "model",
                rr.Transform3D(
                    rotation_axis_angle=rr.RotationAxisAngle(
                        [1.0, 0.0, 0.0], degrees=90.0
                    )
                ),
            )

    def step(self, action: int) -> tuple[Data, float, bool, bool, dict[str, Any]]:
        config = self.place_configs[action - len(self.model)]
        self.engine.place_part(config)
        self.timer += 1
        done = self.timer == self.max_steps
        obs = self.gen_obs()
        if self.use_potential:
            new_score, d = self.score_fn(self.model, obs)
            done = done or d
            reward = new_score - self.last_score
            self.last_score = new_score
        else:
            # Reward at end
            reward = 0.0
            new_score, d = self.score_fn(self.model, obs)
            done = done or d
            if done:
                reward = new_score
        return obs, reward, done, False, {}

    def render(self):
        translations = [[] for _ in range(len(BLOCK_PARTS))]
        rotations = [[] for _ in range(len(BLOCK_PARTS))]
        for placed in self.model:
            pos = placed.position
            translations[placed.part_id].append([pos.x, pos.y, pos.z])
            rot = placed.rotation
            rotations[placed.part_id].append([rot.x, rot.y, rot.z, rot.w])
        for i, model in enumerate(self.models):
            model.render(translations[i], rotations[i])

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Data, dict[str, Any]]:
        self.engine.clear_model()
        self.start_fn(self.engine)
        self.timer = 0
        obs = self.gen_obs()
        if self.use_potential:
            self.last_score, _ = self.score_fn(self.model, obs)
        return obs, {}

    def gen_obs(self) -> Data:
        self.model = self.engine.get_model()
        self.place_configs = self.engine.gen_candidates()
        part_ids = []
        nodes = []
        edges = set()
        for j, config in enumerate(self.model + self.place_configs):
            part_ids.append(config.part_id)
            min_bbox, max_bbox = merge_bboxes(config.bboxes)
            node_vec = torch.zeros([NODE_DIM])
            node_vec[0] = min_bbox[0] / 10.0
            node_vec[1] = min_bbox[1] / 10.0
            node_vec[2] = min_bbox[2] / 10.0
            node_vec[3 + 0] = max_bbox[0] / 10.0
            node_vec[3 + 1] = max_bbox[1] / 10.0
            node_vec[3 + 2] = max_bbox[2] / 10.0
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
        mask_arr = [1] * len(self.model) + [0] * len(self.place_configs)
        if len(mask_arr) >= MAX_NODES:
            raise RuntimeError(
                f"Too many nodes, got {len(mask_arr)} nodes, max is {MAX_NODES}"
            )
        data = Data(
            x=x,
            edge_index=edge_index,
            part_ids=torch.tensor(part_ids),
            action_mask=torch.tensor(mask_arr),
        )
        return data


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
