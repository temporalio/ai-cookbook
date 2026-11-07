"""Workflow and activity payloads.

Pydantic v2 models, serialized by the OpenAI Agents plugin's data converter. One rule keeps them durable: never mutate a
model that is workflow state in place; reassign it with `model_copy(update=...)` (see `AgentState.with_*`).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Grant(Strict):
    """An operator's decision on an egress the sandbox policy denied, recorded in the workflow. Approved grants are
    applied to the running sandbox and re-applied to every sandbox generation built afterwards."""

    host: str
    port: int
    rule_name: str
    decision: Literal["approved", "rejected"]
    approver: str
    at: str
    applied: bool


class GrantDecision(Strict):
    host: str
    decision: Literal["approved", "rejected"]
    approver: str
    notes: str


class ReviewDecision(Strict):
    review_id: str
    action: Literal["approve", "revise", "reject"]
    notes: str


class Artifact(Strict):
    """A file the agent wrote in the exec sandbox, copied to the host as soon as it exists. The sandbox is disposable;
    the artifact is not."""

    name: str
    path: str
    sha256: str
    step: int


class AgentState(Strict):
    run_id: str
    gen: int  # sandbox generation; 0 before the first one is built
    exec_sandbox: str | None
    tools_sandbox: str | None
    step: int
    grants: list[Grant]
    artifacts: list[Artifact]

    def with_grant(self, grant: Grant) -> "AgentState":
        return self.model_copy(update={"grants": [*self.grants, grant]})

    def with_artifacts(self, artifacts: list[Artifact], step: int) -> "AgentState":
        kept = [a for a in self.artifacts if a.name not in {n.name for n in artifacts}]
        return self.model_copy(update={"artifacts": [*kept, *artifacts], "step": step})

    def with_sandboxes(self, gen: int, exec_sandbox: str | None, tools_sandbox: str | None) -> "AgentState":
        return self.model_copy(update={"gen": gen, "exec_sandbox": exec_sandbox, "tools_sandbox": tools_sandbox})


class AgentInput(Strict):
    brief: str
    model: str
    review_timeout_hours: float


class AgentResult(Strict):
    run_id: str
    status: Literal["approved", "rejected"]
    rounds: int
    gen: int
    grants: list[Grant]
    artifacts: list[Artifact]
    decisions: list[dict[str, Any]]
    run_dir: str


# ---------------------------------------------------------------- activity inputs and outputs


class EnsureInput(Strict):
    run_id: str
    workflow_id: str
    gen: int
    grants: list[Grant]
    artifacts: list[Artifact]


class SandboxesRef(Strict):
    gen: int
    exec_sandbox: str
    tools_sandbox: str
    created: list[str]
    rehydrated: list[str]
    retired: list[str]


class SearchInput(Strict):
    run_id: str
    tools_sandbox: str
    host: str
    query: str


class SearchOutput(Strict):
    denied: bool
    results: list[dict[str, Any]]
    detail: str


class RunCodeInput(Strict):
    run_id: str
    exec_sandbox: str
    step: int
    code: str


class RunCodeOutput(Strict):
    ok: bool
    exit_code: int
    stdout_tail: str
    stderr_tail: str
    artifacts: list[Artifact]


class GrantInput(Strict):
    run_id: str
    tools_sandbox: str
    grant: Grant


class ReleaseInput(Strict):
    run_id: str
    gen: int
    exec_sandbox: str | None
    tools_sandbox: str | None


class ReleaseOutput(Strict):
    deleted: list[str]
    logs: list[str]
