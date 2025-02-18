from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel
from kitbasher.env import BLOCK_CONNECT_RULES, BLOCK_PARTS
from kitbasher.utils import parse_args
from kitbasher_rust import EngineWrapper, PyPlacedConfig, Renderer, PartReference


class Config(BaseModel):
    model: str


if __name__ == "__main__":
    cfg: Config = parse_args(Config)

    base_path = "../kitbasher-game/assets/models/{name}.ron"
    ldraw_to_part = {
        "3005": ("1x1", (0.0, 3.0, 0.0), (0, 0, 0)),
        "3040b": ("2x1_slanted", (0.0, 3.0, 2.5), (0, 2, 0)),
        "3004": ("2x1", (0.0, 3.0, 0.0), (0, 1, 0)),
        "30000": ("2x2_axle", (0.0, 3.0, 0.0), (0, 0, 0)),
        "3039": ("2x2_slanted", (0.0, 3.0, 2.5), (0, 2, 0)),
        "3003": ("2x2", (0.0, 3.0, 0.0), (0, 0, 0)),
        "3010": ("4x1", (0.0, 3.0, 0.0), (0, 1, 0)),
        "72206p01": ("wheel", (0.0, 0.0, 0.0), (0, 1, 0)),
    }
    ref_map = {
        k: PartReference(
            BLOCK_PARTS.index(base_path.format(name=part_name)),
            x,
            y,
            z,
            x_rot,
            y_rot,
            z_rot,
        )
        for k, (part_name, (x, y, z), (x_rot, y_rot, z_rot)) in ldraw_to_part.items()
    }
    engine = EngineWrapper(BLOCK_PARTS, BLOCK_CONNECT_RULES, False)
    engine.load_ldraw(cfg.model, ref_map, True)
    engine.shuffle_model_parts()
    model = engine.get_model()

    # Construct trajectory
    parts = list[PyPlacedConfig]()
    for _ in range(1, len(model)):
        parts.append(engine.pop_part())

    # Render model, step by step
    renderer = Renderer(
        [part[: part.rindex(".")] + ".glb" for part in BLOCK_PARTS], True
    )
    while len(parts) > 0:
        part = parts.pop()
        engine.place_part(part)
        model = engine.get_model()
        buffers = renderer.render_model(model)
        front, back = tuple(
            np.array(b).reshape([512, 512, 4])[::-1, :, :3] for b in buffers
        )
        plt.imshow(front)
        plt.show()
