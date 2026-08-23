from typing import Literal

from pydantic import BaseModel

StageName = Literal["analysis", "critique", "recommendation"]


class FixedFlowInput(BaseModel):
    """Source material and model configuration for the fixed flow."""

    topic: str
    source_material: str
    model: str = "gpt-4o-mini"


class AgentStageRequest(BaseModel):
    """Instructions and context for one specialist stage."""

    stage: StageName
    instructions: str
    input: str
    model: str


class AgentStageResult(BaseModel):
    """Text produced by one specialist stage."""

    stage: StageName
    output: str


class FixedFlowResult(BaseModel):
    """Outputs from every stage in the order they ran."""

    analysis: str
    critique: str
    recommendation: str
