"""The agent loop as a Temporal Workflow.

round := build this round's sandboxes (generation n, with every grant and every layer the Workflow recorded) → the agent
runs until it asks for review → the sandboxes are released (gateway logs exported, sandboxes deleted) → wait for a human
decision (a durable timer; nothing is held meanwhile) → approve, reject, or revise and go round again. At the end: the
Relay scopes close and the manifest joins the four records.

Human decisions are Workflow Updates with validators: `grant` (an egress the sandbox policy denied) and `review`.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError
from temporalio.workflow import ActivityCancellationType

with workflow.unsafe.imports_passed_through():
    from agents import Agent, RunConfig, Runner
    from agents.agent import StopAtTools
    from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
    from agents.run_config import ToolExecutionConfig

    from activities import gpu
    from activities import sandbox as sb
    from models import (
        AgentInput,
        AgentResult,
        AgentState,
        CloseInput,
        EnsureInput,
        GrantDecision,
        ManifestInput,
        ReleaseInput,
        ReviewDecision,
    )
    from provider import nemotron_settings
    from sceneops.cheatsheet import CHEATSHEET
    from sceneops.hashing import layer_stack_hash
    from workflows.tools import SANDBOX_OPTS, STEP_CAP, TOOLS, AgentCtx

INSTRUCTIONS = f"""You are a scene-authoring agent for an OpenUSD warehouse digital twin. You work in small, durable steps.
Loop: 1) inspect_site once (docks, lanes, keep-clear zone, build-area hint, what is already placed), 2) search_assets for catalog ids,
3) apply_usd_code with Python that uses the helper API `h`, 4) read placed_assets / agent_prims / violations / warnings,
5) check_lane('L2') and render_preview('Station') when the layout looks complete, 6) fix problems with another apply_usd_code,
7) STOP RULE: when violations are empty, check_lane passed, and every item in the brief is placed, your next call MUST be request_review.
Never call more than one tool per turn. Budget: at most {STEP_CAP} apply_usd_code calls per round, so plan the WHOLE cell in your first apply:
read build_area_aabb and the asset sizes, use h.find_spot(id, yaw_deg=...) / h.conflicts_for(...) inside the code to pick free spots, place every
required item with an explicit name, then use later applies only to fix reported violations or overlaps (h.fix_all / h.set_position / h.set_rotation).
Only catalog ids are placeable (the `id` values from search_assets); never pass file paths or URLs to h.place_asset.
Every required item in the brief (stations, conveyor, pallets, light) must be a real asset or light: floor markings are decoration, not placeholders.
inspect_site is free; apply_usd_code that only prints is rejected. Give prims explicit names and re-place with the same name to move them
(a name replaces the older prim); h.clear() removes everything. Check agent_prims so nothing stale is left in a lane.
Code rules: plain Python, no markdown fences, prefer single-quoted strings, no imports; `h`, `stage`, `Usd`, `Sdf`, `UsdGeom`, `Gf` exist.
If search_assets says denied, say so in one sentence and continue with the catalog ids it returned.
Place assets on open floor inside the build area near the named dock, at least the required clearance from every lane and keep_clear volume,
never overlapping each other, and inside the building.

{CHEATSHEET}"""

RELEASE_OPTS: dict[str, Any] = dict(SANDBOX_OPTS, cancellation_type=ActivityCancellationType.ABANDON)
# A cancelled build reports back before the Workflow releases its sandboxes by name, so nothing it was creating survives.
ENSURE_OPTS: dict[str, Any] = dict(SANDBOX_OPTS, cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED)


def _turn_prompt(inp: AgentInput, state: AgentState, notes: list[str]) -> str:
    lines = [f"Brief: {inp.brief}"]
    if state.layers:
        lines.append(f"Progress so far: {len(state.layers)} layers applied (steps {[layer.step for layer in state.layers]}); last summary: {state.layers[-1].summary}")
    if state.lane_checks:
        lc = state.lane_checks[-1]
        lines.append(f"Last lane check: lane {lc.lane} {'PASSED' if lc.passed else 'FAILED'} (min clearance {lc.min_clearance_m} m).")
    if notes:
        lines.append("Reviewer revisions (treat as the top constraint, newest last): " + " | ".join(notes))
    return "\n".join(lines)


@workflow.defn
class SimulationAgentWorkflow:
    @workflow.init
    def __init__(self, inp: AgentInput) -> None:
        wf_id = workflow.info().workflow_id
        self.state = AgentState(run_id=wf_id, gen=0, exec_sandbox=None, tools_sandbox=None, step=0, grants=[], layers=[], renders=[], lane_checks=[])
        self.ctx: AgentCtx | None = None
        self.round = 0
        self.notes: list[str] = []
        self.decisions: list[dict[str, Any]] = []
        self.pending_review_id: str | None = None
        self.decision: ReviewDecision | None = None
        self.last_apply: dict[str, Any] = {}  # the latest judged layout; inspect_site shows it in every round

    @workflow.run
    async def run(self, inp: AgentInput) -> AgentResult:
        wf_id = workflow.info().workflow_id
        agent = Agent(name="SceneAuthor", instructions=INSTRUCTIONS, tools=TOOLS, model=inp.model, model_settings=nemotron_settings(),
                      tool_use_behavior=StopAtTools(stop_at_tool_names=["request_review"]))
        # An unknown tool name goes back to the model as an error, instead of failing the Workflow Task over and over.
        run_config = RunConfig(tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1), tool_not_found_behavior="return_error_to_model")
        while True:
            self.round += 1
            self.ctx = AgentCtx(run_id=self.state.run_id, workflow_id=wf_id, state=self.state, brief=inp.brief, model=inp.model, gpu_task_queue=inp.gpu_task_queue,
                                rebuild=self._rebuild, round_start_step=self.state.step, last_apply=self.last_apply)
            try:
                summary = await self._round(agent, run_config, inp)
            except BaseException:
                # The round ended with an error or a cancellation: the sandboxes do not outlive it. The release is
                # shielded from the cancellation, and a failure to release must not hide the original error.
                with contextlib.suppress(Exception):
                    await self._release_sandboxes()
                raise
            # Review: nothing is held while a person decides. The sandboxes are gone, their logs are on disk, the layers,
            # the lane checks and the grants are in the Workflow's state.
            await self._release_sandboxes()
            self.pending_review_id = f"{wf_id}:r{self.round}"
            workflow.set_current_details(f"round {self.round}: awaiting review {self.pending_review_id}; sandboxes released. Agent: {summary[:200]}")
            try:
                await workflow.wait_condition(lambda: self.decision is not None, timeout=timedelta(hours=inp.review_timeout_hours), timeout_summary="review window")
            except asyncio.TimeoutError:
                if self.decision is None:  # a decision that arrived with the deadline still counts
                    self.decision = ReviewDecision(review_id=self.pending_review_id, action="reject", notes="review window expired")
            decision, self.decision, self.pending_review_id = self.decision, None, None
            assert decision is not None
            self.decisions.append({"round": self.round, "summary": summary, **decision.model_dump(), "at": workflow.now().isoformat(timespec="seconds")})
            if decision.action in ("approve", "reject"):
                return await self._finish(inp, "approved" if decision.action == "approve" else "rejected")
            self.notes = [*self.notes, decision.notes]  # revise: another round, in a new sandbox generation rebuilt from the recorded layers

    async def _round(self, agent: Agent, run_config: RunConfig, inp: AgentInput) -> str:
        """One round: this round's sandboxes, then the agent's turns until it asks for review. Returns the agent's summary."""
        assert self.ctx is not None
        await self._ensure_sandboxes()
        workflow.set_current_details(f"round {self.round}: agent working, sandbox generation {self.state.gen}, step {self.state.step}")
        prompt = _turn_prompt(inp, self.state, self.notes)
        try:
            for _ in range(3):  # StopAtTools ends the run when request_review is called, even when the tool refused
                self.ctx.review_refusal = None
                try:
                    await Runner.run(agent, prompt, context=self.ctx, max_turns=40, run_config=run_config)
                except ModelBehaviorError as e:  # a malformed model turn: say so and give the agent another go
                    prompt = _turn_prompt(inp, self.ctx.state, self.notes) + f"\nYour last turn was rejected ({str(e)[:200]}). Use only the listed tools."
                    continue
                if self.ctx.review_summary is not None or self.ctx.review_refusal is None:
                    break
                prompt = _turn_prompt(inp, self.ctx.state, self.notes) + "\nYour request_review was refused: " + self.ctx.review_refusal + " Do that now, then request_review again."
        except MaxTurnsExceeded:
            self.ctx.review_summary = self.ctx.review_summary or "Turn budget exhausted; requesting review of the current layout."
        self.state = self.ctx.state
        self.last_apply = self.ctx.last_apply
        return self.ctx.review_summary or "The agent ended its turn without requesting review."

    async def _finish(self, inp: AgentInput, status: str) -> AgentResult:
        """Close the gpu Worker's Relay scope (best effort: the trajectory is snapshotted after every job anyway), then the
        agent Worker writes the manifest, which closes its own scope and joins the four records."""
        wf_id = workflow.info().workflow_id
        try:
            await workflow.execute_activity(gpu.close_trace, CloseInput(run_id=self.state.run_id, reason=status), task_queue=inp.gpu_task_queue, summary="close gpu trace",
                                            schedule_to_close_timeout=timedelta(minutes=2), retry_policy=RetryPolicy(maximum_attempts=1))
        except ActivityError:
            workflow.logger.warning("no gpu Worker closed the Relay scope; its last snapshot stands")
        manifest = await workflow.execute_activity(
            sb.write_manifest, ManifestInput(run_id=self.state.run_id, workflow_id=wf_id, status=status, brief=inp.brief, model=inp.model, state=self.state,
                                             decisions=self.decisions, notes=self.notes), summary="write manifest", **SANDBOX_OPTS)
        workflow.set_current_details(f"{status} after {self.round} round(s); manifest {manifest}")
        return AgentResult(run_id=self.state.run_id, status=status, rounds=self.round, gen=self.state.gen, layer_stack_hash=layer_stack_hash(self.state.layer_pairs()),
                           layers=self.state.layers, grants=self.state.grants, decisions=self.decisions, manifest_path=manifest, run_dir=f"runs/{self.state.run_id}")

    # ------------------------------------------------------------ sandboxes

    def _current(self) -> AgentState:
        return self.ctx.state if self.ctx else self.state

    async def _ensure_sandboxes(self) -> None:
        """A new generation whenever the round starts without sandboxes (the previous ones were released)."""
        base = self._current()
        if not (base.exec_sandbox and base.tools_sandbox):
            await self._build(base.gen + 1)

    async def _rebuild(self) -> None:
        """A tool found its sandbox gone: the next generation, with every grant and every layer the Workflow recorded."""
        await self._build(self._current().gen + 1)

    # @@@SNIPSTART build-generation
    async def _build(self, gen: int) -> None:
        base = self._current()
        wf_id = workflow.info().workflow_id
        try:
            ref = await workflow.execute_activity(
                sb.ensure_sandboxes, EnsureInput(run_id=base.run_id, workflow_id=wf_id, gen=gen, grants=base.grants, layers=base.layers),
                summary=f"sandboxes: generation {gen}" + (f", {len(base.layers)} layers rehydrated" if base.layers else ""), **ENSURE_OPTS)
            if ref.layer_stack_hash != layer_stack_hash(base.layer_pairs()):
                raise RuntimeError(f"generation {gen} recomposed a different layer stack ({ref.layer_stack_hash[:12]} != {layer_stack_hash(base.layer_pairs())[:12]})")
        except BaseException:
            # The build may have created part of generation `gen` before it failed, and state never recorded it. The names
            # are deterministic, so release them by name; a failure to release must not hide the original error.
            with contextlib.suppress(Exception):
                await asyncio.shield(workflow.execute_activity(
                    sb.release_sandboxes, ReleaseInput(run_id=base.run_id, gen=gen, exec_sandbox=sb.sandbox_name(wf_id, "exec", gen), tools_sandbox=sb.sandbox_name(wf_id, "tools", gen)),
                    summary=f"release failed generation {gen}", **RELEASE_OPTS))
            raise
        self.state = base.with_sandboxes(ref.gen, ref.exec_sandbox, ref.tools_sandbox)
        if self.ctx:
            self.ctx.state = self.state
    # @@@SNIPEND

    async def _release_sandboxes(self) -> None:
        """Export the gateway logs and delete this round's sandboxes. Shielded so a Workflow cancellation cannot interrupt
        it, and abandoned rather than cancelled so the Worker finishes the delete even if the Workflow has closed."""
        st = self._current()
        if not (st.exec_sandbox or st.tools_sandbox):
            return
        await asyncio.shield(workflow.execute_activity(
            sb.release_sandboxes, ReleaseInput(run_id=st.run_id, gen=st.gen, exec_sandbox=st.exec_sandbox, tools_sandbox=st.tools_sandbox),
            summary="release sandboxes", **RELEASE_OPTS))
        self.state = st.with_sandboxes(st.gen, None, None)
        if self.ctx:
            self.ctx.state = self.state

    # ------------------------------------------------------------ human decisions

    # @@@SNIPSTART grant-update
    @workflow.update
    async def grant(self, decision: GrantDecision) -> str:
        """The operator's decision on an egress the sandbox policy denied. The tool waiting on it applies it."""
        assert self.ctx is not None
        self.ctx.grant_decision = decision
        return decision.decision

    @grant.validator
    def _validate_grant(self, decision: GrantDecision) -> None:
        pending = self.ctx.pending_grant if self.ctx else None
        if not pending or decision.host != pending["host"] or self.ctx.grant_decision is not None:
            raise ValueError(f"no grant pending for {decision.host!r} (pending: {pending['host'] if pending else None!r})")
    # @@@SNIPEND

    @workflow.update
    async def review(self, decision: ReviewDecision) -> str:
        """The reviewer's decision on the round's result."""
        self.decision = decision
        return decision.action

    @review.validator
    def _validate_review(self, decision: ReviewDecision) -> None:
        if self.pending_review_id is None or decision.review_id != self.pending_review_id or self.decision is not None:
            raise ValueError(f"no matching review pending (pending {self.pending_review_id!r}, got {decision.review_id!r})")

    @workflow.query
    def pending_grant(self) -> dict[str, Any] | None:
        return self.ctx.pending_grant if self.ctx else None

    @workflow.query
    def pending_review(self) -> str | None:
        return self.pending_review_id

    @workflow.query
    def get_state(self) -> AgentState:
        return self._current()
