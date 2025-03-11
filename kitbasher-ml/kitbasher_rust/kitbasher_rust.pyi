from typing import *

class PyVec3:
    x: float
    y: float
    z: float

class PyQuat:
    x: float
    y: float
    z: float
    w: float

class PyAxis:
    X = 0
    Y = 1
    Z = 2

    def __int__(self) -> int: ...

class PyConnector:
    side_a: bool
    axis: PyAxis
    connect_type: int
    position: PyVec3

class PyAABB:
    center: PyVec3
    half_sizes: PyVec3

class PyConnection:
    placed_id: int
    connector_id: int

class PyPlacedConfig:
    position: PyVec3
    part_id: int
    rotation: PyQuat
    connectors: List[PyConnector]
    bboxes: List[PyAABB]
    connections: List[Optional[PyConnection]]

    def to_json(self) -> str: ...
    @staticmethod
    def from_json(s: str) -> PyPlacedConfig: ...

class PartReference:
    def __init__(
        self,
        part_id: int,
        pos_offset_x: float,
        pos_offset_y: float,
        pos_offset_z: float,
        rot_offset_x: int,
        rot_offset_y: int,
        rot_offset_z: int,
    ): ...

class EngineWrapper:
    def __init__(
        self,
        part_paths: List[str],
        connect_rules: List[Tuple[int, int]],
        use_mirror: bool,
    ): ...
    def clear_model(self): ...
    def gen_candidates(self) -> List[PyPlacedConfig]: ...
    def get_model(self) -> List[PyPlacedConfig]: ...
    def set_model(self, model: List[PyPlacedConfig]): ...
    def place_part(self, placement: PyPlacedConfig): ...
    def create_config(
        self, part_id: int, x: float, y: float, z: float
    ) -> PyPlacedConfig: ...
    def load_ldraw(self, path: str, ref_map: Dict[str, PartReference], use_mirror: bool): ...
    def shuffle_model_parts(self): ...
    def pop_part(self) -> PyPlacedConfig: ...

class Renderer:
    def __init__(self, part_paths: List[str], use_mirror: bool): ...
    def render_model(
        self, model: List[PyPlacedConfig]
    ) -> Tuple[List[int], List[int]]: ...
