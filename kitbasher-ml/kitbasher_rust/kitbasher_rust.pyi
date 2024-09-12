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

class EngineWrapper:
    def __init__(self, part_paths: List[str], connect_rules: List[Tuple[int, int]]): ...
    def clear_model(self): ...
    def gen_candidates(self) -> List[PyPlacedConfig]: ...
    def get_model(self) -> List[PyPlacedConfig]: ...
    def place_part(self, placement: PyPlacedConfig): ...
    def create_config(self, part_id: int, x: float, y: float, z: float) -> PyPlacedConfig: ...