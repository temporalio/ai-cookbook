<!--
description: Gate an agent's network access on a human approval that lives in the Workflow, with NVIDIA OpenShell sandboxes you can delete and rebuild with the same grants.
tags: [agents, python, nvidia, openai-agents]
priority: 700
-->

# Durable human approvals for sandboxed agents with NVIDIA OpenShell

An agent that runs code a model wrote needs a sandbox. A sandbox that needs a human to open a door needs that approval
to live somewhere that survives the sandbox, the worker, and a day-long wait. This recipe keeps the approval in the
Temporal Workflow: the sandbox becomes disposable, the approval does not.

- An agent built on the [OpenAI Agents SDK integration](https://docs.temporal.io/develop/python/integrations/openai-agents)
  lays out a warehouse packing cell. [NVIDIA Nemotron](https://build.nvidia.com) answers through NVIDIA NIM, which
  serves the OpenAI chat-completions API, so there is no integration code: a base URL and a key.
- The agent's tools run in two [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) sandboxes. The **exec** sandbox
  runs the code the model wrote and has no network. The **tools** sandbox runs the catalog search and reaches the
  network only through grants an operator approves.
- The first search is denied by policy. The tool returns `denied` instead of raising, publishes a grant request, and the
  Workflow waits on a durable condition. The operator's decision is a **Workflow Update** with a validator. An activity
  applies the grant to the running sandbox (`openshell policy update … --wait`), and the retried search goes through.
- The grant is Workflow state. Delete a sandbox mid-run and the next tool call builds the next generation with the
  same grant and the agent's files put back. Reviews release the sandboxes first, so a review that takes a day holds
  nothing.

The recipe runs without OpenShell too: a fake backend reproduces the control flow in plain directories (no isolation),
which is also what the tests use.

## Layout

| File | What it does |
|---|---|
| `workflows/approval_agent_workflow.py` | The agent loop: sandboxes per round, the agent's turns, the `grant` and `review` Updates and their validators, the review wait |
| `workflows/tools.py` | The agent's tools; `search_assets` holds the deny → ask → apply → retry flow |
| `activities/sandbox.py` | Build a sandbox generation with the recorded grants, run tools in it, apply a grant live, export logs, delete |
| `activities/openshell.py` | The OpenShell CLI adapter and the fake backend |
| `activities/policies/*.yaml` | OpenShell policies: the exec sandbox has no network rules; grants are appended to the tools policy on rebuild |
| `provider.py` | Nemotron on NIM through the OpenAI Agents SDK provider, client retries off |
| `worker.py`, `start_workflow.py` | The worker and a run |
| `grant.py`, `review.py`, `delete_sandbox.py`, `bundle.py` | The operator: decide a grant, decide a review, delete a sandbox, collect the run's records |

## Prerequisites

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- The [Temporal CLI](https://docs.temporal.io/cli) for a local dev server
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) (`NVIDIA_API_KEY`); the hosted NIM endpoints have a free tier
- Optional: [OpenShell](https://docs.nvidia.com/openshell) on Linux (kernel 6.2+) or on Apple Silicon with Docker Desktop.
  Without it, the fake backend is used automatically.

## Run it

```bash
uv sync
export NVIDIA_API_KEY=nvapi-...
temporal server start-dev                     # terminal 1
uv run python -m worker                       # terminal 2 (prints which sandbox backend it uses)
uv run python -m start_workflow               # terminal 3: prints the workflow id
```

Play the operator in terminal 3:

```bash
uv run python -m grant <workflow-id> approved          # the agent's search was denied; approve the host
uv run python -m delete_sandbox <workflow-id> exec     # optional chaos: the next tool call rebuilds generation 2
uv run python -m review <workflow-id> revise --notes "move the pallet 1 m east"   # another round, in a new generation
uv run python -m review <workflow-id> approve
uv run python -m bundle <workflow-id>                  # history, gateway logs, artifacts, state → runs/<workflow-id>/bundle/
```

Watch the run in the Temporal UI (`http://localhost:8233`): one activity per model call, sandbox operation and grant;
the `grant` and `review` Updates in the history; the review wait as a timer while no sandbox exists.

To use real OpenShell sandboxes, install OpenShell, run `openshell status`, and start the worker with the binary on
`PATH` (or `SANDBOX_BACKEND=openshell`). The sandboxes are created from `python:3.12-slim`; set `SANDBOX_IMAGE` and
`SANDBOX_PYTHON` (the interpreter's real path, which the network rules match) to use another image. To follow the
gateway's decisions live: `openshell logs <tools-sandbox> --source sandbox --tail`.

## The approval, in code

The search tool asks for the grant, waits, applies it, and records it. Nothing runs on the Workflow's behalf while it
waits, so the workers can scale to zero.

*File: workflows/tools.py*

```python
@function_tool(failure_error_function=tool_error_to_json)
async def search_assets(rc: RunContextWrapper[AgentCtx], query: str) -> str:
    ctx = rc.context
    search = lambda: workflow.execute_activity(
        sb.sandbox_search, SearchInput(run_id=ctx.run_id, tools_sandbox=ctx.state.tools_sandbox or "", host=USD_SEARCH_HOST, query=query),
        summary=f"search: {query[:40]}", **SANDBOX_OPTS)
    out = await _with_rebuild(ctx, search)
    if not out.denied:
        return json.dumps({"ok": True, "source": "usd_search", "assets": _names(out.results)})

    # Denied by the sandbox policy: ask a person, and park the workflow until they answer.
    ctx.pending_grant = {"host": USD_SEARCH_HOST, "port": 443, "rule_name": "usd_search", "reason": f"SimReady search for: {query[:80]}"}
    await workflow.wait_condition(lambda: ctx.grant_decision is not None, timeout=GRANT_WAIT)
    decision, ctx.grant_decision, ctx.pending_grant = ctx.grant_decision, None, None

    grant = Grant(host=USD_SEARCH_HOST, port=443, rule_name="usd_search", decision=decision.decision, approver=decision.approver,
                  at=workflow.now().isoformat(timespec="seconds"), applied=False)
    if decision.decision == "approved":
        # A live policy update on the running tools sandbox, then the search is retried: allowed this time.
        grant = await _with_rebuild(ctx, lambda: workflow.execute_activity(
            sb.apply_grant, GrantInput(run_id=ctx.run_id, tools_sandbox=ctx.state.tools_sandbox or "", grant=grant), summary="apply grant", **SANDBOX_OPTS))
        ctx.state = ctx.state.with_grant(grant)  # workflow state: every sandbox generation built from now on gets this grant
        out = await _with_rebuild(ctx, search)
        return json.dumps({"ok": True, "denied": out.denied, "operator": "approved", "assets": _names(out.results) if not out.denied else LOCAL_CATALOG})
    ctx.state = ctx.state.with_grant(grant)  # a rejection is recorded too: the agent will not ask again this run
    return json.dumps({"ok": True, "denied": True, "operator": "rejected", "assets": LOCAL_CATALOG})
```

The person's decision is a Workflow Update. The validator runs first; a decision for the wrong host, or a second
decision, never reaches the history.

*File: workflows/approval_agent_workflow.py*

```python
@workflow.update
async def grant(self, decision: GrantDecision) -> str:
    self.ctx.grant_decision = decision
    return decision.decision

@grant.validator
def _validate_grant(self, decision: GrantDecision) -> None:
    pending = self.ctx.pending_grant if self.ctx else None
    if not pending or decision.host != pending["host"] or self.ctx.grant_decision is not None:
        raise ValueError(f"no grant pending for {decision.host!r}")
```

The activity that applies it is the documented live update, and it is idempotent: re-adding the same endpoint is
accepted, so a retried activity is safe.

*File: activities/openshell.py*

```python
async def apply_grant(self, name: str, grant: Grant) -> None:
    r = await self._run("policy", "update", name, "--rule-name", grant.rule_name, "--binary", self.python,
                        "--add-endpoint", f"{grant.host}:{grant.port}:read-write:rest:enforce", "--wait", timeout=120)
```

## Rebuilding without losing control

Every sandbox generation is built from the base policy plus the approved grants in the Workflow's state, and the agent's
files are uploaded back into the new exec sandbox. A tool that finds its sandbox gone rebuilds once and retries.

*File: activities/openshell.py*

```python
def policy_file(self, kind: str, grants: list[Grant]) -> Path:
    text = (POLICIES / f"{kind}.policy.yaml").read_text()
    approved = [g for g in grants if g.decision == "approved"] if kind == "tools" else []   # the exec sandbox never gets network rules
    ...
    for g in approved:
        lines += [f"  {g.rule_name}:", f"    name: {g.rule_name}", "    endpoints:", f"      - host: {g.host}", f"        port: {g.port}",
                  "        protocol: rest", "        access: read-write", "        enforcement: enforce", "    binaries:", f"      - path: {self.python}"]
```

*File: workflows/tools.py*

```python
async def _with_rebuild(ctx: AgentCtx, call: Callable[[], Awaitable[Any]]) -> Any:
    try:
        return await call()
    except ActivityError as e:
        if _cause_type(e) != "SandboxLost":
            raise
        await ctx.rebuild()   # the next generation, with every grant the workflow recorded and every artifact
        return await call()
```

Before each review the Workflow exports every sandbox's gateway log (`openshell logs`) and deletes the sandbox, so
the record of what the sandbox tried to reach outlives the sandbox. `bundle.py` collects the Event History, the logs,
the artifacts and the state into one directory; the run id and the sandbox names join them.

## Tests

```bash
uv run pytest tests/
```

The tests use the fake backend and a scripted model (`temporalio.openai_agents.testing`): the denied search, the
validator rejecting a wrong host and a stale review id, the grant applied and recorded, a revision in a new generation
built with the grant and the rehydrated artifact, and a sandbox deleted mid-round.

## Where this comes from

This recipe is the approval and rebuild pattern of a joint NVIDIA and Temporal demo in which a Nemotron agent lays out
a packing cell in an NVIDIA Isaac Sim warehouse, with NVIDIA NeMo Relay tracing every model and tool call and a
simulated robot judging the result. The blog post describing the full reference architecture is forthcoming on the
NVIDIA Technical Blog.
