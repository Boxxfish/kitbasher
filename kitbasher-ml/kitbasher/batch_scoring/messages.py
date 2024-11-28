from typing import Literal
from pydantic import BaseModel

class ScorerMessage(BaseModel):
    buffer_idx: int
    label_idx: int
    prompts: list[str]
    images: list[str]
    scorer_fn: Literal["clip", "contrastive_clip"]

class ScoredMessage(BaseModel):
    buffer_idx: int
    score: bool