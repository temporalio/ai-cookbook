"""The agent's tools. Function tools run inside the Workflow: they call Activities, reassign Workflow state and turn
Activity failures into JSON the model can act on. The human-approval flow lives in `search_assets`; the OpenUSD authoring
loop in `apply_usd_code`; previews and lane checks go to the `gpu` Task Queue."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Awaitable, Callable

from agents import RunContextWrapper, function_tool
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError
from temporalio.exceptions import TimeoutError as TemporalTimeoutError

with workflow.unsafe.imports_passed_through():
    from activities import gpu
    from activities import sandbox as sb
    from models import (
        AgentState,
        ApplyInput,
        Grant,
        GrantDecision,
        GrantInput,
        RenderInput,
        SearchInput,
        SimulateInput,
    )
    from site_facts import (
        BUILD_AREA,
        BUILD_AREA_HINT,
        BUILDING,
        CAMERAS,
        DOCKS,
        LANES,
        MIN_CLEARANCE_M,
        SITE_VOLUMES,
    )

USD_SEARCH_HOST = "search.simready.omniverse.nvidia.com"  # NVIDIA's public SimReady asset search
GRANT_WAIT = timedelta(minutes=10)  # how long the agent waits for a person before it carries on with the catalog
STEP_CAP = 8  # applies per round; the reviewer's revision starts a new round
# RenderFailed is not listed: the renderer says per error whether a retry can help (a 503 can, an unknown op cannot).
NON_RETRYABLE = ["SandboxLost", "LayerMissing", "RehydrateMismatch", "LintRejected"]
SANDBOX_OPTS: dict[str, Any] = dict(
    start_to_close_timeout=timedelta(minutes=10),
    heartbeat_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=3, non_retryable_error_types=NON_RETRYABLE),
)


def gpu_opts(ctx: "AgentCtx") -> dict[str, Any]:
    """Previews and lane checks run on the gpu Task Queue: the Worker next to the renderer, or the CPU stand-in."""
    return dict(task_queue=ctx.gpu_task_queue, start_to_close_timeout=timedelta(minutes=35), heartbeat_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3, non_retryable_error_types=NON_RETRYABLE))


@dataclass
class AgentCtx:
    """What the tools share with the Workflow during one round. `state` is reassigned, never mutated."""

    run_id: str
    workflow_id: str
    state: AgentState
    brief: str
    model: str
    gpu_task_queue: str
    rebuild: Callable[[], Awaitable[None]]  # set by the Workflow: build the next sandbox generation with every recorded grant and layer
    round_start_step: int = 0
    last_apply: dict[str, Any] = field(default_factory=dict)
    pending_grant: dict[str, Any] | None = None
    grant_decision: GrantDecision | None = None
    review_summary: str | None = None
    review_refusal: str | None = None


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


# @@@SNIPSTART with-rebuild
async def _with_rebuild(ctx: AgentCtx, call: Callable[[], Awaitable[Any]]) -> Any:
    """Run a sandbox Activity. If the sandbox is gone (somebody deleted it), build the next generation, with every grant
    and every layer the Workflow recorded, and retry once."""
    try:
        return await call()
    except ActivityError as e:
        if _cause_type(e) != "SandboxLost":
            raise
        await ctx.rebuild()
        return await call()
# @@@SNIPEND


@function_tool(failure_error_function=tool_error_to_json)
async def inspect_site(rc: RunContextWrapper[AgentCtx]) -> str:
    """Docks, lane and keep-clear volumes, the build area, cameras, what is placed so far and the current violations. Runs no code."""
    ctx = rc.context
    st = ctx.state
    return json.dumps({"docks": DOCKS, "site_volumes": SITE_VOLUMES, "lanes": LANES, "build_area_aabb": BUILD_AREA, "build_area_hint": BUILD_AREA_HINT,
                       "building_aabb": BUILDING, "cameras": CAMERAS, "min_clearance_m": MIN_CLEARANCE_M,
                       "layers_applied": [layer.step for layer in st.layers], "last_layer_summary": st.layers[-1].summary if st.layers else None,
                       "placed_assets": ctx.last_apply.get("placed_assets", []), "agent_prims": ctx.last_apply.get("agent_prims", []),
                       "violations": ctx.last_apply.get("violations", []), "asset_overlaps": ctx.last_apply.get("asset_overlaps", []),
                       "lane_checks": [{"lane": c.lane, "passed": c.passed, "min_clearance_m": c.min_clearance_m, "method": c.method} for c in st.lane_checks[-2:]],
                       "note": "This is the current placement. Do not use apply_usd_code just to print or verify."})


def _ids(results: list[dict[str, Any]]) -> list[str]:
    return [r["id"] for r in results if r.get("id")][:8]


# @@@SNIPSTART search-assets
@function_tool(failure_error_function=tool_error_to_json)
async def search_assets(rc: RunContextWrapper[AgentCtx], query: str) -> str:
    """Search for warehouse assets (USD Search when the sandbox policy allows it, the local catalog otherwise). Returns placeable catalog ids."""
    ctx = rc.context
    search = lambda: workflow.execute_activity(  # noqa: E731 — re-evaluated after a rebuild so it sees the new sandbox name
        sb.sandbox_search, SearchInput(run_id=ctx.run_id, tools_sandbox=ctx.state.tools_sandbox or "", host=USD_SEARCH_HOST, query=query),
        summary=f"search: {query[:40]}", **SANDBOX_OPTS)
    out = await _with_rebuild(ctx, search)
    if not out.denied:
        return json.dumps({"ok": True, "source": out.source, "placeable_ids": _ids(out.results), "results": out.results[:8]})

    prior = [g for g in ctx.state.grants if g.host == USD_SEARCH_HOST]
    if prior:  # the operator already decided for this host in this run: do not ask again
        return json.dumps({"ok": True, "source": out.source, "denied": True, "operator": prior[-1].decision, "placeable_ids": _ids(out.results), "results": out.results[:8]})

    # Denied by the sandbox policy: ask a person, and park the Workflow until they answer. Nothing runs on its behalf
    # while it waits, so the Workers can scale to zero.
    ctx.pending_grant = {"host": USD_SEARCH_HOST, "port": 443, "rule_name": "usd_search", "reason": f"SimReady search for: {query[:80]}"}
    ctx.grant_decision = None
    workflow.logger.info("egress to %s denied; waiting for an operator grant", USD_SEARCH_HOST)
    try:
        await workflow.wait_condition(lambda: ctx.grant_decision is not None, timeout=GRANT_WAIT)
    except asyncio.TimeoutError:
        pass  # a decision that arrived with the deadline still counts
    if ctx.grant_decision is None:
        ctx.pending_grant = None
        return json.dumps({"ok": True, "source": out.source, "denied": True, "operator": "no decision within the grant window", "placeable_ids": _ids(out.results),
                           "results": out.results[:8]})
    decision, ctx.grant_decision, ctx.pending_grant = ctx.grant_decision, None, None

    grant = Grant(host=USD_SEARCH_HOST, port=443, rule_name="usd_search", decision=decision.decision, approver=decision.approver,
                  at=workflow.now().isoformat(timespec="seconds"), applied=False)
    if decision.decision == "approved":
        # A live policy update on the running tools sandbox, then the search is retried: allowed this time.
        grant = await _with_rebuild(ctx, lambda: workflow.execute_activity(
            sb.apply_grant, GrantInput(run_id=ctx.run_id, tools_sandbox=ctx.state.tools_sandbox or "", grant=grant), summary="apply grant: usd_search", **SANDBOX_OPTS))
        ctx.state = ctx.state.with_grant(grant)  # Workflow state: every sandbox generation built from now on gets this grant
        out = await _with_rebuild(ctx, search)
        return json.dumps({"ok": True, "source": out.source, "denied": out.denied, "operator": "approved", "placeable_ids": _ids(out.results), "results": out.results[:8]})
    ctx.state = ctx.state.with_grant(grant)  # a rejection is recorded too: the agent will not ask again this run
    return json.dumps({"ok": True, "source": out.source, "denied": True, "operator": "rejected", "operator_notes": decision.notes, "placeable_ids": _ids(out.results),
                       "results": out.results[:8]})
# @@@SNIPEND


_EDIT_CALL = re.compile(r"\bh\.(place_asset|move|set_position|set_rotation|remove|clear|add_floor_marking|add_light|nudge_clear|fix_all)\s*\(")
_PATH_AS_ID = re.compile(r"h\.place_asset\(\s*[\"'][^\"']*/[^\"']*\.usd[ac]?[\"']")


# @@@SNIPSTART apply-usd-code
@function_tool(failure_error_function=tool_error_to_json)
async def apply_usd_code(rc: RunContextWrapper[AgentCtx], code: str, rationale: str) -> str:
    """Run Python that edits the scene through the helper API `h` inside the exec sandbox. Each call writes one OpenUSD sublayer
    for the step. Returns placed assets, violations (trusted check), overlaps, warnings and stdout_tail."""
    ctx = rc.context
    if ctx.state.step - ctx.round_start_step >= STEP_CAP:
        return json.dumps({"ok": False, "type": "StepCapReached", "msg": f"{STEP_CAP} applies used this round; call check_lane('L2') and render_preview('Station'), then request_review"})
    if not _EDIT_CALL.search(code):
        return json.dumps({"ok": False, "type": "NoEdits", "msg": "This code only prints or inspects. Use inspect_site (free) for the current placement; apply_usd_code "
                           "must call h.place_asset / h.move / h.set_position / h.set_rotation / h.remove / h.clear / h.add_floor_marking / h.add_light / h.fix_all."})
    if _PATH_AS_ID.search(code):
        return json.dumps({"ok": False, "type": "NotACatalogId", "msg": "h.place_asset takes catalog ids only (for example 'packing_table'), never file paths or URLs. "
                           "Use the ids returned by search_assets."})
    step = ctx.state.step + 1
    try:
        out = await _with_rebuild(ctx, lambda: workflow.execute_activity(
            sb.sandbox_apply, ApplyInput(run_id=ctx.run_id, workflow_id=ctx.workflow_id, exec_sandbox=ctx.state.exec_sandbox or "", step=step, code=code,
                                         rationale=rationale, model=ctx.model, prior_layers=ctx.state.layers, min_clearance_m=MIN_CLEARANCE_M),
            summary=f"step {step}: {rationale[:36]}", **SANDBOX_OPTS))
    except ActivityError as e:
        if _cause_type(e) == "SandboxCodeTimeout":
            await ctx.rebuild()  # retire the exec sandbox in case the remote command is still running
        raise
    if out.ok and out.layer is not None:
        ctx.state = ctx.state.with_layer(out.layer)  # the layer is the checkpoint: a rebuilt sandbox starts from every layer recorded here
        ctx.last_apply = {"placed_assets": out.placed_assets, "agent_prims": out.agent_prims, "violations": out.violations, "asset_overlaps": out.asset_overlaps}
    hint = None
    if out.ok and not out.violations and not out.asset_overlaps and out.placed_assets:
        hint = "No violations. If every item in the brief is placed, call check_lane('L2') and render_preview('Station') once, then request_review. Do not add extras."
    elif out.ok and (out.violations or out.asset_overlaps):
        hint = ("Fix the listed violations and overlaps with ONE apply_usd_code: `moves = h.fix_all(); print(moves)` nudges every conflicting asset the least "
                "distance that clears lanes, keep-clear and overlaps, and removes floor markings painted inside a lane (redraw them outside).")
    return json.dumps({"ok": out.ok, "step": step if out.ok else None, "cached": out.cached, "error": out.error, "placed_assets": out.placed_assets,
                       "agent_prims": out.agent_prims, "violations": out.violations, "asset_overlaps": out.asset_overlaps, "warnings": out.warnings,
                       "stdout_tail": out.stdout_tail[-800:], "applies_left": max(0, STEP_CAP - (step - ctx.round_start_step)) if out.ok else None, "hint": hint})
# @@@SNIPEND


@function_tool(failure_error_function=tool_error_to_json)
async def render_preview(rc: RunContextWrapper[AgentCtx], camera: str) -> str:
    """Render the current layer stack from a named camera (Station, Wide, TopDown, LaneL2, RobotEye): Isaac Sim RTX, or a plan view without a GPU."""
    ctx = rc.context
    if not ctx.state.layers:
        return json.dumps({"ok": False, "type": "NothingToRender", "msg": "apply at least one step first"})
    cam = camera.rsplit("/", 1)[-1]
    if cam not in CAMERAS:
        return json.dumps({"ok": False, "type": "UnknownCamera", "msg": f"cameras: {CAMERAS}"})
    ref = await workflow.execute_activity(gpu.render_preview, RenderInput(run_id=ctx.run_id, step=ctx.state.step, layers=ctx.state.layers, camera=cam, width=1280, height=720),
                                          summary=f"step {ctx.state.step}: render {cam}", **gpu_opts(ctx))
    ctx.state = ctx.state.with_render(ref)
    return json.dumps({"ok": True, "camera": cam, "mode": ref.mode, "path": ref.path})


@function_tool(failure_error_function=tool_error_to_json)
async def check_lane(rc: RunContextWrapper[AgentCtx], lane: str) -> str:
    """Simulate the AMR driving the named lane (L1, L2, L3) through the current layout: pass/fail, min clearance, contacts, goal reached."""
    ctx = rc.context
    if not ctx.state.layers:
        return json.dumps({"ok": False, "type": "NothingToCheck", "msg": "apply at least one step first"})
    if lane not in LANES:
        return json.dumps({"ok": False, "type": "UnknownLane", "msg": f"lanes: {LANES}"})
    chk = await workflow.execute_activity(gpu.simulate_lane, SimulateInput(run_id=ctx.run_id, step=ctx.state.step, layers=ctx.state.layers, lane=lane, robot="carter",
                                                                          min_clearance_m=MIN_CLEARANCE_M), summary=f"step {ctx.state.step}: lane {lane}", **gpu_opts(ctx))
    ctx.state = ctx.state.with_lane_check(chk)
    return json.dumps({"ok": True, "lane": lane, "passed": chk.passed, "min_clearance_m": chk.min_clearance_m, "contacts": chk.contacts, "amr_reached_goal": chk.reached,
                       "method": chk.method, "hint": None if chk.passed else "Move the listed assets away from the lane (h.fix_all() or h.set_position), then check again."})


@function_tool(failure_error_function=tool_error_to_json)
async def request_review(rc: RunContextWrapper[AgentCtx], summary: str) -> str:
    """Finish this round and hand the layout to a human reviewer. Call it after check_lane and render_preview on the final layout."""
    ctx = rc.context
    if not ctx.state.layers:
        ctx.review_refusal = "place the cell with apply_usd_code, then run check_lane('L2') and render_preview('Station') on it."
        return json.dumps({"ok": False, "type": "NothingToReview", "msg": "Before review, " + ctx.review_refusal})
    step = ctx.state.step  # the checks must be of the layout the reviewer will see, not of an earlier step
    missing = []
    if ctx.state.current_lane_check("L2") is None:
        missing.append("check_lane('L2')")
    if not any(r.step == step for r in ctx.state.renders):
        missing.append("render_preview('Station')")
    if missing:
        ctx.review_refusal = "run " + " and ".join(missing) + f" on the current layout (step {step}) first (they are not applies)."
        return json.dumps({"ok": False, "type": "ChecksMissing", "msg": "Before review, " + ctx.review_refusal})
    ctx.review_summary = summary
    return "Review requested."


TOOLS = [inspect_site, search_assets, apply_usd_code, render_preview, check_lane, request_review]
