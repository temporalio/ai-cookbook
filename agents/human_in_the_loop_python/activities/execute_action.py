from temporalio import activity
from models.models import ProposedAction
import asyncio

@activity.defn
async def execute_action(action: ProposedAction) -> str:
    """Execute the approved action.
    
    In a real system, this would call external APIs, modify databases,
    or trigger other workflows based on the action_type.
    """
    activity.logger.info(
        f"Executing action: {action.action_type}",
        extra={"action": action.model_dump()}
    )
    
    return f"Successfully executed: {action.action_type}"
