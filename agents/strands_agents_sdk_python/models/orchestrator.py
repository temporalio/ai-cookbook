from pydantic import BaseModel
from typing import List, Optional


class ToolCall(BaseModel):
    tool_name: str
    parameters: dict = {}


class AgentResponse(BaseModel):
    tool_calls: List[ToolCall] = []
    final_answer: Optional[str] = None
    reasoning: Optional[str] = None
