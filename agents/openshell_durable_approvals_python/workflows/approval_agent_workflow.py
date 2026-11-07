"""The agent loop as a Temporal workflow.

round := build this round's sandboxes (generation n, with every grant the workflow recorded) → the agent runs until it
asks for review → the sandboxes are released (gateway logs exported, sandboxes deleted) → wait for a human decision (a
durable timer; nothing is held meanwhile) → approve, reject, or revise and go round again.

Human decisions are Workflow Updates with validators: `grant` (an egress the sandbox policy denied) and `review`.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from agents import Agent, RunConfig, Runner
    from agents.agent import StopAtTools
    from agents.exceptions import MaxTurnsExceeded
    from agents.run_config import ToolExecutionConfig

    from activities import sandbox as sb
    from models import (
        AgentInput,
        AgentResult,
        AgentState,
        EnsureInput,
        GrantDecision,
        ReleaseInput,
        ReviewDecision,
    )
    from provider import nemotron_settings
    from workflows.tools import SANDBOX_OPTS, TOOLS, AgentCtx

INSTRUCTIONS = """You lay out a small warehouse packing cell and write it to a file, in small durable steps.
Tools: search_assets (the asset catalog), run_code (Python in a sandbox with no network; write files into the current
directory), request_review (hand the result to a human).
Steps: 1) search_assets once for the assets the brief needs. If the result says denied, use the asset names it returns;
do not search again. 2) run_code once to write layout.json: a JSON list of placements, each {"name", "asset", "x_m", "y_m"},
with at least 0.6 m between items. 3) request_review with a one-sentence summary.
One tool call per turn. Plain Python only, no markdown fences. Use only the asset names the search returned."""


def _turn_prompt(inp: AgentInput, state: AgentState, notes: list[str]) -> str:
    prompt = f"Brief: {inp.brief}"
    if state.artifacts:
        prompt += "\nFiles you wrote in earlier rounds are in your current directory: " + ", ".join(a.name for a in state.artifacts)
    if notes:
        prompt += "\nReviewer notes, newest last: " + " | ".join(notes)
    return prompt


@workflow.defn
class ApprovalAgentWorkflow:
    @workflow.init
    def __init__(self, inp: AgentInput) -> None:
        self.state = AgentState(run_id=workflow.info().workflow_id, gen=0, exec_sandbox=None, tools_sandbox=None, step=0, grants=[], artifacts=[])
        self.ctx: AgentCtx | None = None
        self.round = 0
        self.notes: list[str] = []
        self.decisions: list[dict[str, Any]] = []
        self.pending_review_id: str | None = None
        self.decision: ReviewDecision | None = None

    @workflow.run
    async def run(self, inp: AgentInput) -> AgentResult:
        wf_id = workflow.info().workflow_id
        agent = Agent(name="PackingCellAgent", instructions=INSTRUCTIONS, tools=TOOLS, model=inp.model, model_settings=nemotron_settings(),
                      tool_use_behavior=StopAtTools(stop_at_tool_names=["request_review"]))
        run_config = RunConfig(tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1))
        while True:
            self.round += 1
            self.ctx = AgentCtx(run_id=self.state.run_id, workflow_id=wf_id, state=self.state, rebuild=self._rebuild)
            await self._ensure_sandboxes()
            workflow.set_current_details(f"round {self.round}: agent working, sandbox generation {self.state.gen}")
            prompt = _turn_prompt(inp, self.state, self.notes)
            try:
                for _ in range(3):  # StopAtTools ends the run when request_review is called, even when the tool refused
                    await Runner.run(agent, prompt, context=self.ctx, max_turns=20, run_config=run_config)
                    if self.ctx.review_summary is not None:
                        break
                    prompt = _turn_prompt(inp, self.ctx.state, self.notes) + "\nYour request_review was refused: write layout.json with run_code first, then request_review again."
            except MaxTurnsExceeded:
                self.ctx.review_summary = "Turn budget exhausted; requesting review of what is there."
            self.state = self.ctx.state
            summary = self.ctx.review_summary or "The agent ended its turn without requesting review."

            # Review: nothing is held while a person decides. The sandboxes are gone, their logs are on disk, the
            # artifacts and the grants are in the workflow's state.
            await self._release_sandboxes()
            self.pending_review_id = f"{wf_id}:r{self.round}"
            workflow.set_current_details(f"round {self.round}: awaiting review {self.pending_review_id}; sandboxes released. Agent: {summary[:200]}")
            try:
                await workflow.wait_condition(lambda: self.decision is not None, timeout=timedelta(hours=inp.review_timeout_hours), timeout_summary="review window")
            except asyncio.TimeoutError:
                self.decision = ReviewDecision(review_id=self.pending_review_id, action="reject", notes="review window expired")
            decision, self.decision, self.pending_review_id = self.decision, None, None
            assert decision is not None
            self.decisions.append({"round": self.round, "summary": summary, **decision.model_dump(), "at": workflow.now().isoformat(timespec="seconds")})
            if decision.action in ("approve", "reject"):
                status = "approved" if decision.action == "approve" else "rejected"
                workflow.set_current_details(f"{status} after {self.round} round(s)")
                return AgentResult(run_id=self.state.run_id, status=status, rounds=self.round, gen=self.state.gen, grants=self.state.grants,
                                   artifacts=self.state.artifacts, decisions=self.decisions, run_dir=f"runs/{self.state.run_id}")
            self.notes = [*self.notes, decision.notes]  # revise: another round, in a new sandbox generation

    # ------------------------------------------------------------ sandboxes

    def _current(self) -> AgentState:
        return self.ctx.state if self.ctx else self.state

    async def _ensure_sandboxes(self) -> None:
        """A new generation whenever the round starts without sandboxes (the previous ones were released)."""
        base = self._current()
        if not (base.exec_sandbox and base.tools_sandbox):
            await self._build(base.gen + 1)

    async def _rebuild(self) -> None:
        """A tool found its sandbox gone: the next generation, with every grant the workflow recorded and every artifact."""
        await self._build(self._current().gen + 1)

    async def _build(self, gen: int) -> None:
        base = self._current()
        ref = await workflow.execute_activity(
            sb.ensure_sandboxes, EnsureInput(run_id=base.run_id, workflow_id=workflow.info().workflow_id, gen=gen, grants=base.grants, artifacts=base.artifacts),
            summary=f"sandboxes: generation {gen}", **SANDBOX_OPTS)
        self.state = base.with_sandboxes(ref.gen, ref.exec_sandbox, ref.tools_sandbox)
        if self.ctx:
            self.ctx.state = self.state

    async def _release_sandboxes(self) -> None:
        if not (self.state.exec_sandbox or self.state.tools_sandbox):
            return
        await workflow.execute_activity(
            sb.release_sandboxes, ReleaseInput(run_id=self.state.run_id, gen=self.state.gen, exec_sandbox=self.state.exec_sandbox, tools_sandbox=self.state.tools_sandbox),
            summary="release sandboxes", **SANDBOX_OPTS)
        self.state = self.state.with_sandboxes(self.state.gen, None, None)
        if self.ctx:
            self.ctx.state = self.state

    # ------------------------------------------------------------ human decisions

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
