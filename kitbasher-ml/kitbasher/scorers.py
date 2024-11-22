import random
from typing import *

from torch_geometric.data import Data

from kitbasher_rust import EngineWrapper, PyPlacedConfig

from kitbasher.env import ConstructionEnv

def single_start(engine: EngineWrapper):
    config = engine.create_config(5, 0, 0, 0)
    engine.place_part(config)


def volume_fill_scorer(
    model: List[PyPlacedConfig], data: Data, env: "ConstructionEnv", is_done: bool
) -> tuple[float, bool]:
    """
    Grants a reward of 1 for every part that touches the volume.
    """
    cx, cy, cz = [0.0, 0.0, 0.0]
    hx, hy, hz = [40.0, 10.0, 5.0]
    score = 0
    for placed in model:
        part_score = 1
        for bbox in placed.bboxes:
            if (
                abs(placed.position.x + bbox.center.x - cx) > (hx + bbox.half_sizes.x)
                or abs(placed.position.y + bbox.center.y - cy)
                > (hy + bbox.half_sizes.y)
                or abs(placed.position.z + bbox.center.z - cz)
                > (hz + bbox.half_sizes.z)
            ):
                part_score = 0
                break
        score += part_score
    return score, False


def connect_start(engine: EngineWrapper):
    # Generate a model with 20 pieces
    parts: List[PyPlacedConfig] = []
    config = engine.create_config(5, 0, 0, 0)
    engine.place_part(config)
    parts.append(config)
    for _ in range(20):
        candidates = engine.gen_candidates()
        config = random.choice(candidates)
        parts.append(config)
        engine.place_part(config)

    engine.clear_model()

    # Place two random parts of the model
    indices = list(range(0, len(parts)))
    random.shuffle(indices)
    for i in range(2):
        part = parts[indices[i]]
        part.connections = [None for _ in part.connections]
        engine.place_part(part)


def connect_scorer(
    model: List[PyPlacedConfig], data: Data, env: "ConstructionEnv", is_done: bool
) -> tuple[float, bool]:
    """
    Returns 1 and ends the episode if the model is connected.
    """
    # Set up adjacency list
    edges: Dict[int, List[int]] = {}
    for e1, e2 in data.edge_index.T.tolist():
        if e1 >= len(model) or e2 >= len(model):
            continue
        if e1 not in edges:
            edges[e1] = []
        if e2 not in edges:
            edges[e2] = []
        edges[e1].append(e2)
        edges[e2].append(e1)

    # Perform DFS to check number of nodes reachable from node 0
    seen = set()
    stack = [0]
    while len(stack) > 0:
        node_idx = stack.pop()
        if node_idx in edges:
            for neighbor in edges[node_idx]:
                if neighbor not in seen:
                    stack.append(neighbor)
            seen.add(node_idx)

    connected = len(seen) == len(model)

    return 1.0 if connected else 0.0, connected


def create_clip_scorer(model_url: str = "openai/clip-vit-base-patch32"):
    from transformers import CLIPProcessor, CLIPModel

    clip = CLIPModel.from_pretrained(model_url)
    processor = CLIPProcessor.from_pretrained(model_url)

    def clip_scorer(
        model: List[PyPlacedConfig], data: Data, env: "ConstructionEnv", is_done: bool
    ) -> tuple[float, bool]:
        """
        Returns the score returned by CLIP.
        """
        if not is_done:
            return 0.0, False
        prompt = env.prompt
        imgs = env.screenshot()
        inputs = processor(
            text=[prompt],
            images=imgs,
            return_tensors="pt",
            padding=True,
        )

        outputs = clip(**inputs)
        logits_per_image = outputs.logits_per_image
        score = logits_per_image.mean().item() / 30.0
        return score, False

    return clip_scorer

def create_contrastive_clip_scorer(model_url: str = "openai/clip-vit-base-patch32"):
    from transformers import CLIPProcessor, CLIPModel

    clip = CLIPModel.from_pretrained(model_url)
    processor = CLIPProcessor.from_pretrained(model_url)

    def clip_scorer(
        model: List[PyPlacedConfig], data: Data, env: "ConstructionEnv", is_done: bool
    ) -> tuple[float, bool]:
        """
        Returns the score returned by CLIP.
        """
        if not is_done:
            return 0.0, False
        label_idx = env.label_idx
        imgs = env.screenshot()
        inputs = processor(
            text=env.prompts,
            images=imgs,
            return_tensors="pt",
            padding=True,
        )

        outputs = clip(**inputs)
        logits_per_image = outputs.logits_per_image.softmax(1)
        score = logits_per_image.mean(0)[label_idx].item()
        return score, False

    return clip_scorer