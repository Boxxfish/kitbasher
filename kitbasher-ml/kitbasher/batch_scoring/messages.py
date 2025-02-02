from typing import Any, Literal
from pydantic import BaseModel
class RenderMessage(BaseModel):
    buffer_idx: int
    label_idx: int
    traj_id: int
    prompts: list[str]
    part_configs: list[str]
    scorer_fn: Literal["clip", "contrastive_clip"]

class ScorerMessage(BaseModel):
    buffer_idx: int
    label_idx: int
    traj_id: int
    prompts: list[str]
    images: list[str]
    scorer_fn: Literal["clip", "contrastive_clip"]

class ScoredMessage(BaseModel):
    buffer_idx: int
    traj_id: int
    score: float