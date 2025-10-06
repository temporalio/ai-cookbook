from pydantic import BaseModel

class ProposedAction(BaseModel):
    """Action proposed by the AI agent for human review."""
    action_type: str
    description: str
    reasoning: str

class ApprovalRequest(BaseModel):
    """Request sent to human reviewer."""
    request_id: str
    proposed_action: ProposedAction
    context: str
    requested_at: str

class ApprovalDecision(BaseModel):
    """Decision received from human reviewer."""
    request_id: str
    approved: bool
    reviewer_notes: str | None = None
    decided_at: str
