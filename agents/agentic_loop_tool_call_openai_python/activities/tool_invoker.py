from temporalio import activity
from typing import Sequence
from temporalio.common import RawValue
import inspect
from pydantic import BaseModel

# We use dynamic activities to allow the agent to be defined independently of the tools it can call.
@activity.defn(dynamic=True)
async def dynamic_tool_activity(args: Sequence[RawValue]) -> dict:
    from tools import get_handler

    # the name of the tool to execute - this is passed in via the execute_activity call in the workflow
    tool_name = activity.info().activity_type 
    tool_args = activity.payload_converter().from_payload(args[0].payload, dict)
    activity.logger.info(f"Running dynamic tool '{tool_name}' with args: {tool_args}")

    handler = get_handler(tool_name)
    # in dynamic activity
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) == 0:
        call_args = []
    else:
        ann = params[0].annotation
        if isinstance(tool_args, dict) and isinstance(ann, type) and issubclass(ann, BaseModel):
            call_args = [ann(**tool_args)]  # or ann.model_validate(tool_args) on Pydantic v2
        else:
            call_args = [tool_args]

    result = await handler(*call_args) if inspect.iscoroutinefunction(handler) else handler(*call_args)

    # Optionally log or augment the result
    activity.logger.info(f"Tool '{tool_name}' result: {result}")
    return result
