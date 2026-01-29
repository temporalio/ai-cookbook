from temporalio import activity
from models.models import ApprovalRequest

@activity.defn
async def notify_approval_needed(request: ApprovalRequest) -> None:
    """Notify external systems that human approval is needed.

    In this sample, the notification comes in the form of messages printed in the terminal running the worker.

    In a real system, this would send notifications via:
    - Email
    - Slack/Teams messages
    - Push notifications to mobile apps
    - Updates to approval queue UI
    """
    # Get workflow ID from activity context
    workflow_id = activity.info().workflow_id

    activity.logger.info(
        f"Approval needed for request: {request.request_id}",
        extra={"request": request.model_dump()}
    )

    # In a real implementation, you would call notification services here
    risk_indicator = "⚠️  RISKY ACTION" if request.proposed_action.risky_action else "✓ Safe Action"
    print(f"\n{'='*60}")
    print(f"APPROVAL REQUIRED")
    print(f"{'='*60}")
    print(f"Workflow ID: {workflow_id}")
    print(f"Request ID: {request.request_id}")
    print(f"Risk Level: {risk_indicator}")
    print(f"Action: {request.proposed_action.action_type}")
    print(f"Description: {request.proposed_action.description}")
    print(f"Reasoning: {request.proposed_action.reasoning}")
    print(f"\nTo approve or reject, use the send_approval script:")
    print(f"  uv run python -m send_approval {workflow_id} {request.request_id} approve 'Looks good'")
    print(f"  uv run python -m send_approval {workflow_id} {request.request_id} reject 'Too risky'")
    print(f"{'='*60}\n")
