"""activity_tool wrapper for ADK + Temporal.

Thin adapter over ``temporalio.google_adk.workflow.activity_as_tool`` that
adds **graceful failure**: when an activity execution fails (retry policy
exhausted, non-retryable application error, timeout, etc.), the
``ActivityError`` is caught and returned as a string to the LLM instead
of crashing the ADK pipeline. This lets agents reason about tool
failures — e.g. the Dispatch Agent can still produce an assignment when
the Fleet Agent's tool is down. Temporal still records the retry
attempts in workflow history.

Only ``ActivityError`` is caught. Programming bugs (e.g. argument-binding
``TypeError`` from the upstream adapter, or other bugs in this wrapper)
propagate normally so they surface clearly instead of being silently
re-presented to the LLM.

The upstream ``activity_as_tool`` already handles multi-arg activities and
local (non-workflow) ADK runs; this wrapper only layers the
error-to-tool-response behavior on top.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from temporalio.exceptions import ActivityError
from temporalio.google_adk.workflow import activity_as_tool


def activity_tool(activity_def: Callable, **kwargs: Any) -> Callable:
    """Wrap a Temporal Activity as an ADK Tool with graceful failure.

    Delegates to the upstream adapter for arg binding and activity execution,
    then catches ``ActivityError`` and returns it to the LLM as a string.
    """
    inner = activity_as_tool(activity_def, **kwargs)

    # functools.wraps copies the name, docstring, annotations and the
    # ``__signature__`` the upstream adapter set, so ADK builds the tool
    # schema from the activity's own parameters rather than ``*args, **kw``.
    @functools.wraps(inner)
    async def wrapper(*args: Any, **kw: Any):
        try:
            return await inner(*args, **kw)
        except ActivityError as e:
            return f"ERROR: Tool {activity_def.__name__} failed: {e}"

    return wrapper
