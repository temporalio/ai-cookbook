# ABOUTME: Dynamic Temporal activity for executing tool functions.
# Uses Temporal's dynamic activity feature to execute any registered tool by name.

import inspect
from collections.abc import Sequence

from pydantic import BaseModel
from temporalio import activity
from temporalio.common import RawValue


@activity.defn(dynamic=True)
async def dynamic_tool_activity(args: Sequence[RawValue]) -> dict:
    """Execute a tool dynamically based on the activity name.

    This activity uses Temporal's dynamic activity feature. The activity name
    (passed via execute_activity) becomes the tool name, allowing tools to be
    added/removed without changing the workflow code.
    """
    from tools import get_handler

    # The tool name comes from the activity type (how it was invoked)
    tool_name = activity.info().activity_type
    tool_args = activity.payload_converter().from_payload(args[0].payload, dict)
    activity.logger.info(f"Running dynamic tool '{tool_name}' with args: {tool_args}")

    handler = get_handler(tool_name)

    # Inspect the handler's signature to determine how to pass arguments
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) == 0:
        call_args = []
    else:
        ann = params[0].annotation
        if isinstance(tool_args, dict) and isinstance(ann, type) and issubclass(ann, BaseModel):
            # Handler expects a Pydantic model - instantiate it
            call_args = [ann(**tool_args)]
        else:
            call_args = [tool_args]

    if not inspect.iscoroutinefunction(handler):
        raise TypeError("Tool handler must be async (awaitable).")

    result = await handler(*call_args)
    activity.logger.info(f"Tool '{tool_name}' result: {result}")
    return result
