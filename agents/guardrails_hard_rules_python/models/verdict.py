from typing import Literal
from pydantic import BaseModel, Field


class LLMVerdict(BaseModel):
    """The LLM's raw classification — used as the tool output schema."""
    classification: Literal["safe", "review", "block"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Verdict(LLMVerdict):
    """Final verdict after hard rules have been applied."""
    overridden_by_hard_rule: bool = False
