from pydantic import BaseModel
from typing import List, Dict, Any


class AgentRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"


class WeatherRequest(BaseModel):
    city: str