"""activity_tool wrapper for ADK + Temporal.

Thin adapter over ``temporalio.contrib.google_adk_agents.workflow.activity_tool``
that adds **graceful failure**: when an activity exhausts its retry policy,
the error is caught and returned as a string to the LLM instead of crashing
the ADK pipeline. This lets agents reason about tool failures — e.g. the
Dispatch Agent can still produce an assignment when the Fleet Agent's tool
is down. Temporal still records the retry attempts in workflow history.

The upstream ``activity_tool`` (temporalio>=1.25) already handles multi-arg
activities and local (non-workflow) ADK runs; this wrapper only layers the
error-to-tool-response behavior on top.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from temporalio.contrib.google_adk_agents.workflow import (
    activity_tool as _upstream_activity_tool,
)


def activity_tool(activity_def: Callable, **kwargs: Any) -> Callable:
    """Wrap a Temporal Activity as an ADK Tool with graceful failure.

    Delegates to the upstream adapter for arg binding and activity execution,
    then catches retry-exhaustion (and any other exception) and returns it to
    the LLM as a string. ADK's tool schema generation still sees the original
    activity signature.
    """
    inner = _upstream_activity_tool(activity_def, **kwargs)

    async def wrapper(*args: Any, **kw: Any):
        try:
            return await inner(*args, **kw)
        except Exception as e:
            return f"ERROR: Tool {activity_def.__name__} failed: {e}"

    wrapper.__name__ = activity_def.__name__
    wrapper.__doc__ = activity_def.__doc__
    setattr(wrapper, "__signature__", inspect.signature(activity_def))

    return wrapper
