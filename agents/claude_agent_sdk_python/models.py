"""
Pydantic Models for Agent Execution

Defines the input/output contract between the workflow and activity.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    """Input for the agent execution activity."""

    prompt: str = Field(..., description="User message to send to the agent")
    model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude model to use",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt for the agent",
    )
    max_turns: int = Field(
        default=30,
        description="Maximum number of agentic turns (tool call rounds)",
    )
    permission_mode: str = Field(
        default="bypassPermissions",
        description="Claude Code permission mode (e.g. 'bypassPermissions', 'default')",
    )
    resume_session_id: Optional[str] = Field(
        default=None,
        description="Session ID to resume a previous conversation. "
        "The SDK stores sessions as JSONL files on disk; passing this "
        "resumes from where the last session left off.",
    )


class AgentOutput(BaseModel):
    """Output from the agent execution activity."""

    status: Literal["success", "error"] = Field(
        ..., description="Overall execution status"
    )
    response: str = Field(default="", description="Final assistant response text")
    total_tokens: int = Field(default=0, description="Total tokens used")
    num_events: int = Field(default=0, description="Number of SDK events processed")
    processing_time_seconds: Optional[float] = Field(
        default=None, description="Wall-clock time in seconds"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error details if status is 'error'"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID from the SDK, used to resume this conversation later",
    )
