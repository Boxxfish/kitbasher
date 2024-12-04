from typing import Any, Literal
from pydantic import BaseModel

TO_RENDER_ADDR = "tcp://*:5558"
TO_SCORER_ADDR = "tcp://*:5559"
TO_TRAINER_ADDR = "tcp://*:5560"

class RenderMessage(BaseModel):
    buffer_idx: int
    label_idx: int
    prompts: list[str]
    part_configs: list[Any]
    scorer_fn: Literal["clip", "contrastive_clip"]

class ScorerMessage(BaseModel):
    buffer_idx: int
    label_idx: int
    prompts: list[str]
    images: list[str]
    scorer_fn: Literal["clip", "contrastive_clip"]

class ScoredMessage(BaseModel):
    buffer_idx: int
    score: bool