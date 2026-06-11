"""activity_tool wrapper for ADK + Temporal.

Thin adapter over ``temporalio.contrib.google_adk_agents.workflow.activity_tool``
that adds **graceful failure**: when an activity execution fails (retry
policy exhausted, non-retryable application error, timeout, etc.), the
``ActivityError`` is caught and returned as a string to the LLM instead
of crashing the ADK pipeline. This lets agents reason about tool
failures — e.g. the Dispatch Agent can still produce an assignment when
the Fleet Agent's tool is down. Temporal still records the retry
attempts in workflow history.

Only ``ActivityError`` is caught. Programming bugs (e.g. argument-binding
``TypeError`` from the upstream adapter, or other bugs in this wrapper)
propagate normally so they surface clearly instead of being silently
re-presented to the LLM.

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
from temporalio.exceptions import ActivityError


def activity_tool(activity_def: Callable, **kwargs: Any) -> Callable:
    """Wrap a Temporal Activity as an ADK Tool with graceful failure.

    Delegates to the upstream adapter for arg binding and activity execution,
    then catches ``ActivityError`` and returns it to the LLM as a string.
    ADK's tool schema generation still sees the original activity signature.
    """
    inner = _upstream_activity_tool(activity_def, **kwargs)

    async def wrapper(*args: Any, **kw: Any):
        try:
            return await inner(*args, **kw)
        except ActivityError as e:
            return f"ERROR: Tool {activity_def.__name__} failed: {e}"

    wrapper.__name__ = activity_def.__name__
    wrapper.__doc__ = activity_def.__doc__
    # Copy the activity's annotations too, not just the signature. ADK builds
    # tool schemas via ``typing.get_type_hints(wrapper)``; without this the
    # hints come from ``wrapper(*args, **kw)`` so every real parameter is
    # missing, which raises ``KeyError`` for activities whose module uses
    # ``from __future__ import annotations`` (stringized hints).
    wrapper.__annotations__ = dict(getattr(activity_def, "__annotations__", {}))
    setattr(wrapper, "__signature__", inspect.signature(activity_def))

    return wrapper
