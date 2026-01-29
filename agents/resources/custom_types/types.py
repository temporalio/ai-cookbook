from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class AgentInput(BaseModel):
    task: str


class AgentStepInput(BaseModel):
    """
    Single step input to the LLM activity.
    """

    task: str
    history: List[Dict[str, Any]]


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class AgentStepOutput(BaseModel):
    # Whether the model finished
    is_final: bool

    # Plain-text final answer
    output_text: Optional[str] = None

    # If tool requested:
    tool_call: Optional[ToolCall] = None

    # Raw model message for history (optional)
    model_message: Dict[str, Any]


class ValidateCompanyArgs(BaseModel):
    company_name: str


class IdentifySectorArgs(BaseModel):
    company_name: str


class IdentifyCompetitorsArgs(BaseModel):
    sector: str
    company_name: str


class BrowsePageArgs(BaseModel):
    url: str
    instructions: str


class GenerateReportArgs(BaseModel):
    company_name: str
    context: str
