"""Workflow and Activity payloads.

Pydantic v2 models, serialized by the OpenAI Agents plugin's data converter. Two rules keep them durable: list fields are
required (the converter drops unset fields), and Workflow state is never mutated in place; it is reassigned with
`model_copy(update=...)` (see `AgentState.with_*`). Bytes never travel as payloads: layers, renders, clips and logs are
files under runs/<workflow-id>/, and the payloads carry their paths and hashes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Grant(Strict):
    """An operator's decision on an egress the sandbox policy denied, recorded in the Workflow. Approved grants are
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


class LayerRef(Strict):
    """One OpenUSD sublayer the agent authored in one step: a file under runs/<workflow-id>/layers/ and its hash. The
    sandbox that wrote it is disposable; the layer is the checkpoint."""

    step: int
    path: str
    sha256: str
    bytes: int
    summary: str


class RenderRef(Strict):
    step: int
    camera: str
    mode: str  # rt2: Isaac Sim RTX through the renderer service; plan: the CPU plan view
    path: str
    sha256: str


class LaneCheck(Strict):
    step: int
    lane: str
    passed: bool
    min_clearance_m: float
    contacts: list[str]
    reached: bool | None
    clip_path: str | None
    method: str  # physx: the Isaac Sim drive-through; geometry: footprints against the lane volume, on the CPU


class AgentState(Strict):
    run_id: str
    gen: int  # sandbox generation; 0 before the first one is built
    exec_sandbox: str | None
    tools_sandbox: str | None
    step: int  # the last applied layer's step
    grants: list[Grant]
    layers: list[LayerRef]
    renders: list[RenderRef]
    lane_checks: list[LaneCheck]

    def with_grant(self, grant: Grant) -> "AgentState":
        return self.model_copy(update={"grants": [*self.grants, grant]})

    def with_layer(self, layer: LayerRef) -> "AgentState":
        return self.model_copy(update={"layers": [*self.layers, layer], "step": layer.step})

    def with_render(self, render: RenderRef) -> "AgentState":
        return self.model_copy(update={"renders": [*self.renders, render]})

    def with_lane_check(self, check: LaneCheck) -> "AgentState":
        return self.model_copy(update={"lane_checks": [*self.lane_checks, check]})

    def with_sandboxes(self, gen: int, exec_sandbox: str | None, tools_sandbox: str | None) -> "AgentState":
        return self.model_copy(update={"gen": gen, "exec_sandbox": exec_sandbox, "tools_sandbox": tools_sandbox})

    def layer_pairs(self) -> list[tuple[int, str]]:
        return [(layer.step, layer.sha256) for layer in self.layers]


class AgentInput(Strict):
    brief: str
    model: str
    review_timeout_hours: float
    gpu_task_queue: str  # where previews and lane checks run: the Worker next to the renderer, or the laptop's CPU stand-in


class AgentResult(Strict):
    run_id: str
    status: Literal["approved", "rejected"]
    rounds: int
    gen: int
    layer_stack_hash: str
    layers: list[LayerRef]
    grants: list[Grant]
    decisions: list[dict[str, Any]]
    manifest_path: str
    run_dir: str


# ---------------------------------------------------------------- activity inputs and outputs


class EnsureInput(Strict):
    run_id: str
    workflow_id: str
    temporal_run_id: str  # sandboxes belong to one Workflow Execution, never to the Workflow Id
    gen: int
    grants: list[Grant]
    layers: list[LayerRef]


class SandboxesRef(Strict):
    gen: int
    exec_sandbox: str
    tools_sandbox: str
    created: list[str]
    rehydrated_steps: list[int]
    layer_stack_hash: str
    retired: list[str]


class SearchInput(Strict):
    run_id: str
    tools_sandbox: str
    host: str
    query: str


class SearchOutput(Strict):
    denied: bool
    source: str  # usd_search | catalog
    results: list[dict[str, Any]]
    detail: str


class ApplyInput(Strict):
    run_id: str
    workflow_id: str
    exec_sandbox: str
    step: int
    code: str
    rationale: str
    model: str
    prior_layers: list[LayerRef]
    min_clearance_m: float


class ApplyOutput(Strict):
    ok: bool
    cached: bool
    layer: LayerRef | None
    error: dict[str, Any] | None
    warnings: list[str]
    violations: list[dict[str, Any]]
    asset_overlaps: list[dict[str, Any]]
    placed_assets: list[dict[str, Any]]
    agent_prims: list[dict[str, Any]]
    stdout_tail: str
    layer_stack_hash: str


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


class RenderInput(Strict):
    run_id: str
    step: int
    layers: list[LayerRef]
    camera: str
    width: int
    height: int


class SimulateInput(Strict):
    run_id: str
    step: int
    layers: list[LayerRef]
    lane: str
    robot: str
    min_clearance_m: float


class CloseInput(Strict):
    run_id: str
    reason: str


class ManifestInput(Strict):
    run_id: str
    workflow_id: str
    status: str
    brief: str
    model: str
    state: AgentState
    decisions: list[dict[str, Any]]
    notes: list[str]
