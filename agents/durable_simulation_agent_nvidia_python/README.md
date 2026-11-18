<!--
description: Build a durable, controlled simulation agent in Python with Temporal, NVIDIA OpenShell sandboxes, OpenUSD layers, Isaac Sim checks, Nemotron on NIM, and NeMo Relay traces.
tags: [agents, python, nvidia]
priority: 500
-->

# Durable, controlled simulation agents with NVIDIA OpenShell, Omniverse and NeMo Relay

A simulation agent runs for hours, burns GPU time, executes code a model wrote, and needs a person to approve things
along the way. The hard parts are not the model calls. They are state, retries, human gates, trust boundaries and
provenance. This recipe is the reference demo Temporal built with NVIDIA: the same agent loop, the same tools, the same
records. It runs on a laptop with CPU stand-ins for the GPU parts, and it includes the Isaac Sim service for when you
have a GPU.

- **Temporal remembers.** The agent loop is a Workflow. Every model call, sandbox operation, render and lane check is an
  Activity with retries and heartbeats on one of two Task Queues. Human decisions are Workflow Updates with validators.
  Kill the Worker mid model call and the run resumes from the Event History: only the interrupted model call runs again.
- **NVIDIA OpenShell contains.** The code the model wrote runs in an *exec* sandbox whose policy has no network rules.
  The agent's networked tools run in a *tools* sandbox that reaches the network only through grants an operator approves
  (`openshell policy update … --wait`). Delete a sandbox mid-run and the next tool call rebuilds the next generation with
  every recorded grant and every recorded layer.
- **NVIDIA Omniverse simulates and judges.** The agent authors the warehouse cell as one OpenUSD sublayer per step,
  through a small helper API inside the exec sandbox. Every layer is linted before anything else opens it and judged
  against the trusted site layer. With the recipe's renderer service on a GPU host, Isaac Sim renders previews and
  drives a Nova Carter AMR through the lane; without it, a plan view and a geometry check stand in.
- **NVIDIA Nemotron thinks, NeMo Relay records.** Nemotron answers through NVIDIA NIM. Every Activity and every model
  call is a span in a NeMo Relay trajectory that continues across a Worker restart, with marks for the retry and for the
  fallback model that answered when the primary failed.

At the end, one run leaves four records joined by the Workflow Id: the Event History, the sandboxes' gateway logs, the
Relay trajectories, and the OpenUSD layers with their stack hash. `bundle.py` collects them.

## Layout

| File | What it does |
|---|---|
| `workflows/simulation_agent_workflow.py` | The agent loop: sandboxes per round, the agent's turns, the `grant` and `review` Updates and their validators, the review wait, the manifest |
| `workflows/tools.py` | The agent's tools: `inspect_site`, `search_assets` (the approval flow), `apply_usd_code`, `render_preview`, `check_lane`, `request_review` |
| `activities/sandbox.py` | Build a sandbox generation with the recorded grants and layers, search, apply a step, apply a grant live, export logs, delete, write the manifest |
| `activities/gpu.py` | The `gpu` Task Queue: previews and lane checks, as Isaac Sim jobs or as CPU stand-ins |
| `activities/openshell.py` | The OpenShell CLI adapter and the fake backend |
| `activities/policies/*.yaml` | OpenShell policies: the exec sandbox has no network rules; grants are appended to the tools policy on rebuild |
| `sceneops/` | What runs inside the sandboxes: the helper API the model's code uses, the apply entrypoint, the search entrypoint |
| `validate/` | The host side: the Sdf lint, the trusted judge and lane check, the plan view |
| `site/` | The trusted site layer (docks, lanes, keep-clear volumes, cameras) and the asset catalog with measured footprints |
| `relay.py`, `provider.py` | NeMo Relay: one scope per run per Worker, Activities as tool spans, the model pipeline with fallback |
| `worker.py`, `start_workflow.py` | The Workers and a run |
| `grant.py`, `review.py`, `delete_sandbox.py`, `kill_worker.py`, `bundle.py`, `rehearsal.py` | The operator: decide a grant, decide a review, two chaos beats, collect the records, play every beat unattended |
| `sandbox-image/` | The sandbox image for real OpenShell: Python, usd-core, sceneops and the site |
| `renderer/` | The Isaac Sim service for a GPU host: a job API, the Isaac Sim backend, its image, and an asset mirror |

## Prerequisites

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- The [Temporal CLI](https://docs.temporal.io/cli) for a local Temporal Service
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) (`NVIDIA_API_KEY`); the hosted NIM endpoints have a free tier
- Optional: [OpenShell](https://docs.nvidia.com/openshell) on Linux (kernel 6.2+) or on Apple Silicon with Docker Desktop
- Optional: a Linux host with an NVIDIA RTX GPU, Docker with the NVIDIA Container Toolkit, and an NGC account for the
  Isaac Sim 6.1 container

## Run it

```bash
uv sync
export NVIDIA_API_KEY=nvapi-...
export SANDBOX_BACKEND=fake                   # no OpenShell: directories, no isolation (see below for real sandboxes)
temporal server start-dev                     # terminal 1
uv run worker.py                              # terminal 2: the agent Worker and the gpu Worker in one process
uv run start_workflow.py                      # terminal 3: prints the Workflow Id
```

Export the two variables in terminals 2 and 3: the operator scripts below start the Worker again after a kill, and that
Worker needs the key.

The brief asks for packing cell 2 near Dock 4: two packing stations, a conveyor, pallet staging, floor markings and a
task light, at least 0.6 m from forklift lane L2 and out of the painted keep-clear zone. Play the operator in terminal 3,
in the order the demo plays its beats:

```bash
uv run grant.py <workflow-id> approved          # beat 1: the agent's search was denied; approve the host
uv run delete_sandbox.py <workflow-id> exec     # beat 2: the next tool call rebuilds generation 2 with the grant and the layers
uv run kill_worker.py <workflow-id>             # beat 3: SIGKILL the Worker mid model call; Temporal retries only that Activity
uv run review.py <workflow-id> revise --notes "move the pallet staging 1 m further from lane L2"   # beat 4: another round, generation 3
uv run review.py <workflow-id> approve
uv run bundle.py <workflow-id>                  # beat 5: history, gateway logs, Relay trajectories, layers -> runs/<workflow-id>/bundle/
```

Beats 2 and 3 need the agent at work. With a fast model, round 1 can end seconds after the grant; then play them during
round 2, after the revise. `kill_worker.py` says so when a review is pending and catches the next model call either way.
It kills the Worker that is running the call, found by the identity Temporal reports for it, and starts that Worker again
in the background with the same settings, logging to `runs/_worker.log`.

`uv run rehearsal.py` plays all of it unattended against a running Worker. Watch the run in the Temporal UI
(`http://localhost:8233`): one Activity per model call, sandbox operation, render and lane check, on two Task Queues;
the `grant` and `review` Updates in the Event History; the review wait as a Timer while no sandbox exists. On disk,
`runs/<workflow-id>/` fills up with `layers/NNN.usda`, `renders/`, `lane/`, `logs/`, `relay/` and the `manifest.json`.

To make the fallback model answer, set `FAULT_MODEL_CALLS=3` on the Worker: the third primary model call fails and the
trajectory records a `model_fallback` mark. The count is per Worker process. The defaults are Nemotron 3.5 Lightning
with Nemotron 3.5 Super VL as the fallback; set `NIM_MODEL` and `NIM_FALLBACK_MODEL` for other pairs.

### Real OpenShell sandboxes

Build the sandbox image, push it where your OpenShell provider can pull it, and start the Worker with the binary on
`PATH` and `SANDBOX_BACKEND` unset:

```bash
sandbox-image/build.sh                          # REGISTRY=ghcr.io/<org> sandbox-image/build.sh to push
export SANDBOX_IMAGE=durable-simulation-agent-sandbox:0.1
uv run worker.py
```

The exec sandbox runs `sceneops.apply` with the model's code on stdin and no network; the tools sandbox runs
`sceneops.search`, which the policy denies until the grant. `SANDBOX_PYTHON` is the interpreter's real path inside the
image, which the network rules match. To follow the gateway's decisions live: `openshell logs <tools-sandbox> --source sandbox --tail`.

### Isaac Sim previews and lane checks

`renderer/` is a job service in front of Isaac Sim 6.1 with two operations. `render_still` renders an RTX frame from a
named camera of the site layer. `simulate_lane` checks the lane against the real geometry and drives the Nova Carter
through it with PhysX. On the GPU host, mirror the assets the renderer opens, build and start its container, and start
the gpu Worker next to it; the same tool calls become Isaac Sim jobs:

```bash
uv run renderer/mirror_assets.py --dest /opt/isaac-assets      # the warehouse, the Nova Carter, every catalog asset: about 4.6 GB
docker login nvcr.io                                           # user $oauthtoken, password: your NGC API key
docker build -f renderer/Dockerfile -t durable-simulation-agent-renderer:0.1 .
docker run -d --name renderer --gpus all --network host --user root \
  -v /opt/isaac-assets/Assets/Isaac/6.1:/opt/assets:ro -v /srv/renderer:/srv/renderer \
  durable-simulation-agent-renderer:0.1                        # first start compiles shaders: give it a few minutes
curl -s localhost:8799/healthz                                 # {"backend": "kit", "ready": true, ...}
RENDERER_URL=http://127.0.0.1:8799 RENDERER_SHARED_DIR=/srv/renderer uv run worker.py gpu   # the agent Worker runs anywhere
```

The gpu Worker lints each layer and copies it into the shared directory by content hash; the renderer lints it again
before Kit opens it, and writes its outputs back under the same directory. To try the job API without a GPU:
`uv run python -m renderer.run_service --backend fake --out /tmp/renderer-out`.

Without `RENDERER_URL`, `render_preview` draws a plan view of the composed layers and `check_lane` measures every
footprint against the lane volume. The Workflow sees the same `RenderRef` and `LaneCheck` either way.

Sandboxes belong to one Workflow Execution: their names and labels carry the Run Id. Don't give this Workflow an
execution timeout. A timed-out or terminated execution can't release its sandboxes; the next execution with the same
Workflow Id releases them, keeping their gateway logs, but until then they hold whatever was granted to them.

Layers, renders, lane checks and gateway logs are files under `RUNS_DIR` (default `runs/`) on the Worker host that ran
the Activity. With Workers on more than one host, point `RUNS_DIR` at storage every Worker can reach; otherwise a
rebuild or a render on another host fails with `LayerMissing`.

## The approval, in code

The search tool asks for the grant, waits, applies it, and records it. Nothing runs on the Workflow's behalf while it
waits, so the Workers can scale to zero.

*File: workflows/tools.py*

<!--SNIPSTART workflows/tools.py:search-assets-->
```python
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
```
<!--SNIPEND-->

The person's decision is a Workflow Update. The validator runs first; a decision for the wrong host, or a second
decision, never reaches the Event History.

*File: workflows/simulation_agent_workflow.py*

<!--SNIPSTART workflows/simulation_agent_workflow.py:grant-update-->
```python
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
```
<!--SNIPEND-->

The Activity that applies it is the documented live update, and it is idempotent: re-adding the same endpoint is
accepted, so a retried Activity is safe.

*File: activities/openshell.py*

<!--SNIPSTART activities/openshell.py:apply-grant-->
```python
async def apply_grant(self, name: str, grant: Grant) -> None:
    """The documented live policy update: `openshell policy update <sandbox> --rule-name … --binary … --add-endpoint … --wait`.
    The sandbox logs `CONFIG:LOADED Policy reloaded`; re-adding the same endpoint is accepted, so retries are safe."""
    r = await self._run("policy", "update", name, "--rule-name", grant.rule_name, "--binary", self.python,
                        "--add-endpoint", f"{grant.host}:{grant.port}:read-write:rest:enforce", "--wait", timeout=120)
    if r.exit_code != 0:
        raise (SandboxLost if await self._gone(name, r) else SandboxError)(f"policy update {name}: {r.stderr.strip()[:300]}")
```
<!--SNIPEND-->

## The layers, in code

Each `apply_usd_code` call runs the model's Python through the helper API in the exec sandbox and leaves one sublayer for
the step. The Activity copies the layer out and lints it before anything else opens it. The lint is an allowlist: only
direct children of `/World/Agent`, with the metadata, prim types and API schemas the helper API writes; no variants,
payloads, inherits, value clips, sublayers or scripting attributes anywhere; references only to the catalog's own asset
files. A path under the asset root is not enough, because the root also holds assets that carry OmniGraph nodes; the
catalog's assets carry none, and `renderer/mirror_assets.py` audits that whenever it mirrors them. Then the Activity
judges the layer against the trusted site layer, measuring every asset and piece of geometry under its full transform,
and records the layer's hash in the Workflow's state. The sandbox caches by step and code hash, so a retried Activity
gets the same layer back.

*File: workflows/tools.py*

<!--SNIPSTART workflows/tools.py:apply-usd-code-->
```python
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
    out = await _with_rebuild(ctx, lambda: workflow.execute_activity(
        sb.sandbox_apply, ApplyInput(run_id=ctx.run_id, workflow_id=ctx.workflow_id, exec_sandbox=ctx.state.exec_sandbox or "", step=step, code=code,
                                     rationale=rationale, model=ctx.model, prior_layers=ctx.state.layers, min_clearance_m=MIN_CLEARANCE_M),
        summary=f"step {step}: {rationale[:36]}", **SANDBOX_OPTS))
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
```
<!--SNIPEND-->

Every sandbox generation is built from the base policy plus the approved grants in the Workflow's state, and the
recorded layers are uploaded back into the new exec sandbox and recomposed. The layer stack hash the sandbox reports has
to match the Workflow's record, so a rebuilt generation starts exactly where the deleted one stopped.

*File: workflows/simulation_agent_workflow.py*

<!--SNIPSTART workflows/simulation_agent_workflow.py:build-generation-->
```python
async def _build(self, gen: int) -> None:
    base = self._current()
    wf_id, run_id = workflow.info().workflow_id, workflow.info().run_id
    try:
        ref = await workflow.execute_activity(
            sb.ensure_sandboxes, EnsureInput(run_id=base.run_id, workflow_id=wf_id, temporal_run_id=run_id, gen=gen, grants=base.grants, layers=base.layers),
            summary=f"sandboxes: generation {gen}" + (f", {len(base.layers)} layers rehydrated" if base.layers else ""), **ENSURE_OPTS)
        if ref.layer_stack_hash != layer_stack_hash(base.layer_pairs()):
            raise RuntimeError(f"generation {gen} recomposed a different layer stack ({ref.layer_stack_hash[:12]} != {layer_stack_hash(base.layer_pairs())[:12]})")
    except BaseException:
        # The build may have created part of generation `gen` before it failed, and state never recorded it. The names
        # are deterministic, so release them by name; a failure to release must not hide the original error.
        with contextlib.suppress(Exception):
            await asyncio.shield(workflow.execute_activity(
                sb.release_sandboxes, ReleaseInput(run_id=base.run_id, gen=gen, exec_sandbox=sb.sandbox_name(wf_id, run_id, "exec", gen),
                                                   tools_sandbox=sb.sandbox_name(wf_id, run_id, "tools", gen)),
                summary=f"release failed generation {gen}", **RELEASE_OPTS))
        raise
    self.state = base.with_sandboxes(ref.gen, ref.exec_sandbox, ref.tools_sandbox)
    if self.ctx:
        self.ctx.state = self.state
```
<!--SNIPEND-->

A tool that finds its sandbox gone rebuilds once and retries. Whatever ends a round, a review request, an error, or a
cancellation, the Workflow releases the sandboxes before it moves on.

*File: workflows/tools.py*

<!--SNIPSTART workflows/tools.py:with-rebuild-->
```python
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
```
<!--SNIPEND-->

## The trace, in code

NeMo Relay observes the same run that Temporal makes durable. Each Worker process keeps one Relay agent scope per run
while the run is active; the first one stores the scope's propagation context with the run, and every later process, the
gpu Worker or the agent Worker after a kill, continues the same trace from it. Each run's trajectory keeps only its own
spans, even when one Worker serves many runs. Every Activity is a tool span, with its attempt and a
`temporal_retry` mark when Temporal retried it.

*File: relay.py*

<!--SNIPSTART relay.py:traced-->
```python
def traced(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Activity decorator: the Activity runs as a Relay tool call under the run's agent scope. Relay records the call
    with its input, result, attempt and duration; the body runs in Temporal's own context, so heartbeats, cancellation
    and activity.info() are untouched. A retried attempt also leaves a `temporal_retry` mark."""

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not enabled() or not activity.in_activity():
            return await fn(*args, **kwargs)
        info = activity.info()
        await _close_idle(info.workflow_id)
        run = _open(info.workflow_id)
        run.in_flight += 1
        r = _r()
        with r.use_scope_stack(run.stack):
            if info.attempt > 1:
                r.scope.event("temporal_retry", handle=run.handle, data={"activity": info.activity_type, "attempt": info.attempt})
            th = r.tools.call(info.activity_type, {"activity_id": info.activity_id, "attempt": info.attempt, "input": _jsonable(args[0] if args else None)}, handle=run.handle)
        try:
            out = await fn(*args, **kwargs)
        except BaseException as e:
            with r.use_scope_stack(run.stack):
                r.tools.call_end(th, r.ToolExecutionResult({"error": f"{type(e).__name__}: {str(e)[:200]}"}, {"status": "error"}))
            run.activities += 1
            _after(run)
            raise
        with r.use_scope_stack(run.stack):
            r.tools.call_end(th, r.ToolExecutionResult(_jsonable(out)))
        run.activities += 1
        _after(run)
        return out

    return wrapper
```
<!--SNIPEND-->

Model calls go through Relay's managed LLM pipeline, so the trajectory has the request summary, the response summary and
which model answered. A provider-side failure is routed to the fallback model, with a `model_fallback` mark. Set
`NEMO_RELAY=0` and every hook is a pass-through.

*File: provider.py*

<!--SNIPSTART provider.py:relay-model-->
```python
class RelayModel(Model):
    """Model proxy: the call goes through NeMo Relay's managed LLM pipeline, which records the request summary, the
    response summary and Relay's trace headers; a provider-side failure is answered by the fallback model, with a
    `model_fallback` mark in the trajectory."""

    def __init__(self, inner: Model, name: str, fallback: Model | None, fallback_name: str | None) -> None:
        self._inner, self._name, self._fallback, self._fallback_name = inner, name, fallback, fallback_name

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        kw = _bind(args, kwargs)
        chosen: dict[str, Any] = {"model": self._name, "fallback": False}

        async def call(headers: dict[str, Any], content: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            kw2 = dict(kw)
            forwarded = {k: v for k, v in headers.items() if k in FORWARD_HEADERS and isinstance(v, str)}
            if forwarded:
                kw2["model_settings"] = _with_headers(kw2.get("model_settings"), forwarded)
            try:
                if _fault_due():
                    raise InjectedFault("injected provider fault (FAULT_MODEL_CALLS)")
                resp = await self._inner.get_response(**kw2)
            except Exception as e:  # noqa: BLE001
                if self._fallback is None or not _worth_fallback(e):
                    raise
                err = f"{type(e).__name__}: {str(e)[:120]}"
                chosen.update(model=self._fallback_name, fallback=True, primary=self._name, primary_error=err)
                relay.mark_here("model_fallback", {"primary": self._name, "fallback": self._fallback_name, "error": err})
                if activity.in_activity():
                    activity.logger.warning("model fallback %s -> %s after %s", self._name, self._fallback_name, err)
                resp = await self._fallback.get_response(**kw2)
            return resp, {**chosen, **_response_summary(resp)}

        return await relay.model_call(self._name, _request_summary(kw), call)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.stream_response(*args, **kwargs)
```
<!--SNIPEND-->


## Tests

```bash
uv run pytest tests/
```

The tests use the fake backend, the CPU stand-ins and a scripted model (`temporalio.openai_agents.testing`): the denied
search, the validator rejecting a wrong host and a stale review id, the grant applied and recorded, a layer with a lane
violation and the lane check that fails on it, a revision in a new generation rebuilt from the recorded layer with the
same layer stack hash, the one-call fix, approval and the manifest, the four records on disk; a sandbox deleted
mid-round; a cancellation that still releases the sandboxes; a retried apply that runs the code once; an unknown tool
name that goes back to the model; a review refused until the checks cover the current layout, and an unchecked layout
that can be revised or rejected but not approved. The lint, including the payloads, variants, nested prims and
non-catalog references it refuses, the judge with a scaled crate, an untagged asset and hand-built geometry in a lane, a
re-placed name keeping only its new asset, a damaged apply cache, the search mapping and the site constants are tested
on the host with usd-core. The sandbox Activities are tested for a later execution retiring an earlier one's leftovers
without inheriting its grant, and for a release that deletes both sandboxes when a log or a delete fails. The Relay runtime is tested with two process
identities resuming one trace, two runs in one process keeping separate trajectories, an idle run closed and resumed,
and a faulted model call answered by the fallback. The gpu Activities are tested against the renderer's job service with
a fake backend, including a transient failure that is retried and run again. A stub `openshell` CLI covers the adapter's error classification and that a cancelled call stops the CLI.
