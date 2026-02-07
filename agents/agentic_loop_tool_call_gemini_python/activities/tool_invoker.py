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

    Handles both:
    - Tools with no parameters
    - Tools with Pydantic model parameters (nested LLM output like {'request': {...}})
    """
    from tools import get_handler

    # The tool name comes from the activity type (how it was invoked)
    tool_name = activity.info().activity_type
    tool_args = activity.payload_converter().from_payload(args[0].payload, dict)
    activity.logger.info(f"Running dynamic tool '{tool_name}' with args: {tool_args}")

    handler = get_handler(tool_name)

    if not inspect.iscoroutinefunction(handler):
        raise TypeError("Tool handler must be async (awaitable).")

    # Inspect the handler's signature to determine how to pass arguments
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) == 0:
        # No parameters
        result = await handler()
    else:
        # Get the parameter name and annotation
        param = params[0]
        param_name = param.name
        ann = param.annotation

        if isinstance(ann, type) and issubclass(ann, BaseModel):
            # Handler expects a Pydantic model
            # LLM produces nested output like {'request': {'state': 'CA'}}
            # Extract the nested dict using the parameter name
            nested_args = tool_args.get(param_name, tool_args)
            result = await handler(ann(**nested_args))
        else:
            # Plain parameters - unpack dict as keyword arguments
            result = await handler(**tool_args)

    activity.logger.info(f"Tool '{tool_name}' result: {result}")
    return result
