"""The agent's tools. Function tools run inside the workflow: they call activities, reassign workflow state, and turn
activity failures into JSON the model can act on. The human-approval flow lives in `search_assets`."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Awaitable, Callable

from agents import RunContextWrapper, function_tool
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError
from temporalio.exceptions import TimeoutError as TemporalTimeoutError

with workflow.unsafe.imports_passed_through():
    from activities import sandbox as sb
    from models import (
        AgentState,
        Grant,
        GrantDecision,
        GrantInput,
        RunCodeInput,
        SearchInput,
    )

USD_SEARCH_HOST = "search.simready.omniverse.nvidia.com"  # NVIDIA's public SimReady asset search
GRANT_WAIT = timedelta(minutes=10)  # how long the agent waits for a person before it carries on without the host
LOCAL_CATALOG = ["packing_table", "conveyor_belt", "pallet", "task_light"]  # what the agent can use without the search host
SANDBOX_OPTS: dict[str, Any] = dict(
    start_to_close_timeout=timedelta(minutes=10),
    heartbeat_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=3, non_retryable_error_types=["SandboxLost", "ArtifactMissing"]),
)


@dataclass
class AgentCtx:
    """What the tools share with the workflow during one round. `state` is reassigned, never mutated."""

    run_id: str
    workflow_id: str
    state: AgentState
    rebuild: Callable[[], Awaitable[None]]  # set by the workflow: build the next sandbox generation with every recorded grant
    pending_grant: dict[str, Any] | None = None
    grant_decision: GrantDecision | None = None
    review_summary: str | None = None


def _unwrap(err: BaseException) -> dict[str, Any]:
    cause: BaseException = err
    while isinstance(cause, ActivityError) and cause.cause is not None:
        cause = cause.cause
    if isinstance(cause, ApplicationError):
        return {"ok": False, "type": cause.type or "ApplicationError", "msg": cause.message}
    if isinstance(cause, TemporalTimeoutError):
        return {"ok": False, "type": "ActivityTimeout", "msg": str(cause.type)}
    return {"ok": False, "type": type(cause).__name__, "msg": str(cause)[:300]}


def tool_error_to_json(ctx: Any, err: Exception) -> str | None:
    """`failure_error_function` for @function_tool: the model gets the cause as JSON; cancellation propagates (None re-raises)."""
    if getattr(err, "cancelled_error", None) is not None or isinstance(err, CancelledError):
        return None
    return json.dumps(_unwrap(err))


def _cause_type(e: ActivityError) -> str:
    c: BaseException = e
    while isinstance(c, ActivityError) and c.cause is not None:
        c = c.cause
    return c.type or "" if isinstance(c, ApplicationError) else type(c).__name__


async def _with_rebuild(ctx: AgentCtx, call: Callable[[], Awaitable[Any]]) -> Any:
    """Run a sandbox activity. If the sandbox is gone (somebody deleted it), build the next generation, with every grant the
    workflow recorded, and retry once."""
    try:
        return await call()
    except ActivityError as e:
        if _cause_type(e) != "SandboxLost":
            raise
        await ctx.rebuild()
        return await call()


def _names(results: list[dict[str, Any]]) -> list[str]:
    names = []
    for r in results:
        name = r.get("name") or str(r.get("url", "")).rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if name and name not in names:
            names.append(name)
    return names[:8]


@function_tool(failure_error_function=tool_error_to_json)
async def search_assets(rc: RunContextWrapper[AgentCtx], query: str) -> str:
    """Search the asset catalog for warehouse equipment. Returns asset names to use in the layout."""
    ctx = rc.context
    search = lambda: workflow.execute_activity(  # noqa: E731 — re-evaluated after a rebuild so it sees the new sandbox name
        sb.sandbox_search, SearchInput(run_id=ctx.run_id, tools_sandbox=ctx.state.tools_sandbox or "", host=USD_SEARCH_HOST, query=query),
        summary=f"search: {query[:40]}", **SANDBOX_OPTS)
    out = await _with_rebuild(ctx, search)
    if not out.denied:
        return json.dumps({"ok": True, "source": "usd_search", "assets": _names(out.results)})

    prior = [g for g in ctx.state.grants if g.host == USD_SEARCH_HOST]
    if prior:  # the operator already decided for this host in this run: do not ask again
        return json.dumps({"ok": True, "source": "local_catalog", "denied": True, "operator": prior[-1].decision, "assets": LOCAL_CATALOG})

    # Denied by the sandbox policy: ask a person, and park the workflow until they answer. Nothing runs on its behalf
    # while it waits, so the workers can scale to zero.
    ctx.pending_grant = {"host": USD_SEARCH_HOST, "port": 443, "rule_name": "usd_search", "reason": f"SimReady search for: {query[:80]}"}
    ctx.grant_decision = None
    workflow.logger.info("egress to %s denied; waiting for an operator grant", USD_SEARCH_HOST)
    try:
        await workflow.wait_condition(lambda: ctx.grant_decision is not None, timeout=GRANT_WAIT)
    except asyncio.TimeoutError:
        ctx.pending_grant = None
        return json.dumps({"ok": True, "source": "local_catalog", "denied": True, "operator": "no decision within the grant window", "assets": LOCAL_CATALOG})
    decision, ctx.grant_decision, ctx.pending_grant = ctx.grant_decision, None, None
    assert decision is not None

    grant = Grant(host=USD_SEARCH_HOST, port=443, rule_name="usd_search", decision=decision.decision, approver=decision.approver,
                  at=workflow.now().isoformat(timespec="seconds"), applied=False)
    if decision.decision == "approved":
        # A live policy update on the running tools sandbox, then the search is retried: allowed this time.
        grant = await _with_rebuild(ctx, lambda: workflow.execute_activity(
            sb.apply_grant, GrantInput(run_id=ctx.run_id, tools_sandbox=ctx.state.tools_sandbox or "", grant=grant), summary="apply grant: usd_search", **SANDBOX_OPTS))
        ctx.state = ctx.state.with_grant(grant)  # workflow state: every sandbox generation built from now on gets this grant
        out = await _with_rebuild(ctx, search)
        return json.dumps({"ok": True, "source": "usd_search" if not out.denied else "local_catalog", "denied": out.denied, "operator": "approved",
                           "assets": _names(out.results) if not out.denied else LOCAL_CATALOG})
    ctx.state = ctx.state.with_grant(grant)  # a rejection is recorded too: the agent will not ask again this run
    return json.dumps({"ok": True, "source": "local_catalog", "denied": True, "operator": "rejected", "operator_notes": decision.notes, "assets": LOCAL_CATALOG})


@function_tool(failure_error_function=tool_error_to_json)
async def run_code(rc: RunContextWrapper[AgentCtx], code: str, purpose: str) -> str:
    """Run Python in the exec sandbox, which has no network. Write files into the current directory; every file you write is kept."""
    ctx = rc.context
    step = ctx.state.step + 1
    out = await _with_rebuild(ctx, lambda: workflow.execute_activity(
        sb.sandbox_run_code, RunCodeInput(run_id=ctx.run_id, exec_sandbox=ctx.state.exec_sandbox or "", step=step, code=code),
        summary=f"step {step}: {purpose[:40]}", **SANDBOX_OPTS))
    ctx.state = ctx.state.with_artifacts(out.artifacts, step)  # new or changed files, and the step count
    return json.dumps({"ok": out.ok, "exit_code": out.exit_code, "stdout": out.stdout_tail[-800:], "stderr": out.stderr_tail[-800:],
                       "artifacts": [a.name for a in out.artifacts]})


@function_tool(failure_error_function=tool_error_to_json)
async def request_review(rc: RunContextWrapper[AgentCtx], summary: str) -> str:
    """Finish this round and hand the layout to a human reviewer. Call it once layout.json is written."""
    ctx = rc.context
    if not ctx.state.artifacts:
        return json.dumps({"ok": False, "type": "NothingToReview", "msg": "Write layout.json with run_code first, then request_review."})
    ctx.review_summary = summary
    return "Review requested."


TOOLS = [search_assets, run_code, request_review]
