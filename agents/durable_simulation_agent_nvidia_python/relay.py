"""NeMo Relay (NVIDIA, Apache-2.0) records the same run that Temporal makes durable.

Activity side only. Every Activity of a run, and every model call the Agents plugin makes for it, executes inside one Relay
agent scope per run per Worker process. The scope's propagation context is stored with the run
(runs/<workflow-id>/relay/propagation.json), so another process, the gpu Worker or the agent Worker after a kill, continues
the same trace: the same root id, with a nested "resumed" scope. Each process writes an ATIF trajectory (Relay's
AtifExporter) and a JSONL of lifecycle events into that directory, snapshotted after every Activity, so a killed process
leaves its part of the record behind. Relay's subscribers are process-wide, so each snapshot keeps only the steps of
its own run's scope tree, and a run with no Activity for NEMO_RELAY_IDLE_CLOSE_S seconds (default 300, for example
during a review wait) is closed; its next Activity resumes the trace. Switch off with NEMO_RELAY=0; every hook is then a
pass-through.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable

from temporalio import activity

from paths import run_dir

RUNTIME = "nemo-relay"
AGENT_NAME, AGENT_VERSION = "simulation agent", "0.1"


class _Cfg:
    identity: str = f"{socket.gethostname().split('.')[0]}-{os.getpid()}"


@dataclass
class _Run:
    key: str  # the workflow id
    stack: Any
    handle: Any
    atif: Any
    events_file: Any
    events_path: Path
    resumed: bool
    root_uuid: str
    sub_atif: str
    sub_events: str
    open_n: int  # export file names are unique per open, so a re-open never overwrites an earlier export
    known: set[str]  # uuids of this run's scope tree: the run scope and every span under it
    started: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    in_flight: int = 0  # Activities and model calls of this run running in this process right now; never closed as idle
    activities: int = 0
    llm_calls: int = 0
    fallbacks: int = 0
    closed: bool = False


_runs: dict[str, _Run] = {}
_opened = {"n": 0}


def enabled() -> bool:
    return os.environ.get("NEMO_RELAY", "1") == "1"


def configure(*, identity: str | None = None) -> None:
    """A new process identity (tests play several Workers in one process); open run scopes of this process are dropped."""
    for run in list(_runs.values()):
        _deregister(run)
    _runs.clear()
    if identity:
        _Cfg.identity = identity


def relay_dir(run_id: str) -> Path:
    return run_dir(run_id) / "relay"


def version() -> str:
    try:
        from importlib.metadata import version as v

        return f"{RUNTIME} {v('nemo-relay')}"
    except Exception:  # noqa: BLE001
        return RUNTIME


def _r():
    import nemo_relay  # the native runtime; imported lazily so NEMO_RELAY=0 hosts never load it

    return nemo_relay


def _jsonable(v: Any, depth: int = 0, limit: int = 200) -> Any:
    """Size-bounded JSON view of an Activity input or output: scalars kept, strings trimmed, lists shortened."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit] + "…"
    if depth > 3:
        return str(v)[:limit]
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1, limit) for k, x in list(v.items())[:40]}
    if isinstance(v, (list, tuple, set)):
        v = list(v)
        return [_jsonable(x, depth + 1, limit) for x in v[:20]] + ([f"… +{len(v) - 20}"] if len(v) > 20 else [])
    if hasattr(v, "model_dump"):
        return _jsonable(v.model_dump(), depth + 1, limit)
    return str(v)[:limit]


def _event_writer(path: Path, known: set[str]):
    """A subscriber that writes this run's lifecycle events (the subscriber itself is process-wide: events are kept when
    they belong to the run's scope tree, which this also records in `known`)."""
    f = path.open("a", buffering=1)

    def on_event(ev: Any) -> None:
        uuid, parent = getattr(ev, "uuid", None), getattr(ev, "parent_uuid", None)
        if uuid not in known and parent not in known:
            return
        known.add(uuid)
        try:
            rec = ev.to_dict()
        except Exception:  # noqa: BLE001
            rec = {k: getattr(ev, k, None) for k in ("kind", "name", "uuid", "parent_uuid", "timestamp", "data", "metadata")}
        f.write(json.dumps({"worker": _Cfg.identity, **_jsonable(rec, limit=400)}, default=str) + "\n")

    return f, on_event


def _open(key: str) -> _Run:
    run = _runs.get(key)
    if run:
        return run
    r = _r()
    d = relay_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    prop = d / "propagation.json"
    ctx_json = prop.read_text() if prop.exists() else None
    stack = r.create_scope_stack_from_propagation(r.PropagationContext.from_json(ctx_json)) if ctx_json else r.create_scope_stack()
    _opened["n"] += 1
    n = _opened["n"]
    sub_atif, sub_events = f"atif:{key}:{_Cfg.identity}:{n}", f"events:{key}:{_Cfg.identity}:{n}"
    with r.use_scope_stack(stack):
        atif = r.AtifExporter(f"{key}/{_Cfg.identity}", AGENT_NAME, AGENT_VERSION, model_name=os.environ.get("NIM_MODEL"))
        atif.register(sub_atif)
        handle = r.scope.push(f"run:{key}" + (" (resumed)" if ctx_json else ""), r.ScopeType.Agent,
                              data={"run_id": key, "worker": _Cfg.identity, "resumed": bool(ctx_json),
                                    "temporal_namespace": activity.info().workflow_namespace if activity.in_activity() else None})
        ctx = r.capture_propagation_context()
        if not ctx_json:
            prop.write_text(ctx.to_json())
    known = {ctx.parent_uuid}
    events_file, on_event = _event_writer(d / f"events-{_Cfg.identity}-{n}.jsonl", known)
    r.subscribers.register(sub_events, on_event)
    run = _Run(key, stack, handle, atif, events_file, d / f"events-{_Cfg.identity}-{n}.jsonl", bool(ctx_json), ctx.root_uuid, sub_atif, sub_events, n, known)
    _runs[key] = run
    return run


def _deregister(run: _Run) -> None:
    r = _r()
    for call in (lambda: run.atif.deregister(run.sub_atif), lambda: run.atif.deregister(), lambda: r.subscribers.deregister(run.sub_events), run.events_file.close):
        try:
            call()
        except Exception:  # noqa: BLE001
            pass


def _own(entry: dict[str, Any], known: set[str]) -> bool:
    extra = entry.get("extra") or {}
    return bool({extra.get("event_uuid"), extra.get("parent_event_uuid")} & known)


def _run_trajectory(doc: dict[str, Any], known: set[str]) -> dict[str, Any]:
    """The exporter sees every run in the process. Keep this run's model steps, and this run's tool calls and results
    wherever the exporter attached them; renumber the steps and recompute the totals."""
    steps: list[dict[str, Any]] = []
    for step in doc.get("steps") or []:
        ancestry = (step.get("extra") or {}).get("ancestry") or {}
        own_step = bool({ancestry.get("parent_id"), ancestry.get("function_id")} & known)
        calls = [c for c in step.get("tool_calls") or [] if _own(c, known)]
        observation = step.get("observation") or {}
        results = [res for res in observation.get("results") or [] if _own(res, known)]
        if not own_step and not calls and not results:
            continue
        kept = dict(step) if own_step else {"source": "system", "message": "", "timestamp": step.get("timestamp")}
        kept.pop("tool_calls", None)
        kept.pop("observation", None)
        if calls:
            kept["tool_calls"] = calls
        if results:
            kept["observation"] = {**observation, "results": results}
        steps.append(kept)
    for i, step in enumerate(steps, 1):
        step["step_id"] = i
    metrics = [step.get("metrics") or {} for step in steps if step.get("source") == "agent"]
    doc["steps"] = steps
    doc["final_metrics"] = {"total_prompt_tokens": sum(m.get("prompt_tokens") or 0 for m in metrics),
                            "total_completion_tokens": sum(m.get("completion_tokens") or 0 for m in metrics), "total_steps": len(steps)}
    return doc


def _snapshot(run: _Run) -> Path:
    p = relay_dir(run.key) / f"atif-{_Cfg.identity}-{run.open_n}.json"
    tmp = p.with_name(f".{p.name}.part")
    tmp.write_text(json.dumps(_run_trajectory(json.loads(run.atif.export_json()), run.known), indent=1))
    tmp.replace(p)  # a Worker killed mid-write leaves the previous snapshot, never a truncated one
    return p


async def _close_run(run: _Run, reason: str) -> None:
    """Close the run's scope in this process: a `run_closed` mark, the scope popped, subscribers drained, the final
    snapshot written. A later Activity of the same run in this process resumes the trace from propagation.json."""
    _runs.pop(run.key, None)
    run.closed = True
    r = _r()
    stats = {"reason": reason, "worker": _Cfg.identity, "resumed": run.resumed, "activities": run.activities, "llm_calls": run.llm_calls,
             "fallbacks": run.fallbacks, "seconds": round(time.time() - run.started, 1)}
    with r.use_scope_stack(run.stack):
        r.scope.event("run_closed", handle=run.handle, data=stats)
        r.scope.pop(run.handle, output=stats)
    await r.subscribers.flush_async()  # subscriber delivery is asynchronous; drain before exporting
    _snapshot(run)
    _deregister(run)


async def _close_idle(current: str) -> None:
    """Close the scopes of runs that have had no Activity in this process for a while (a run waiting days for its
    review holds nothing here), so a long-lived Worker keeps only its active runs open."""
    limit = float(os.environ.get("NEMO_RELAY_IDLE_CLOSE_S", "300"))
    now = time.time()
    for run in [r for k, r in _runs.items() if k != current and r.in_flight == 0 and now - r.last_used > limit]:
        try:
            await _close_run(run, "idle")
        except Exception:  # noqa: BLE001 — tracing never fails the Activity that happened to trigger the cleanup
            pass


def _after(run: _Run) -> None:
    run.last_used = time.time()
    run.in_flight = max(0, run.in_flight - 1)
    if run.closed:  # the Activity that closed the run is still inside its own tool span
        return
    try:
        _snapshot(run)
    except Exception:  # noqa: BLE001
        pass


# @@@SNIPSTART traced
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
# @@@SNIPEND


ProviderFn = Callable[[dict[str, Any], dict[str, Any]], Awaitable[tuple[Any, dict[str, Any]]]]


async def model_call(model_name: str, request: dict[str, Any], fn: ProviderFn) -> Any:
    """The model provider's hook (provider.py): the call runs through Relay's managed LLM pipeline (guardrails, request
    intercepts, the LLM span). `fn(headers, content)` makes the real call and returns (response, summary). The pipeline
    only passes JSON back, so the response object travels in a holder and the summary is what Relay records."""
    if not enabled() or not activity.in_activity():
        out, _ = await fn({}, request)
        return out
    info = activity.info()
    await _close_idle(info.workflow_id)
    run = _open(info.workflow_id)
    run.in_flight += 1
    r = _r()
    holder: dict[str, Any] = {}

    async def provider(req: Any) -> Any:
        out, summary = await fn(dict(req.headers), dict(req.content))
        holder["out"] = out
        return summary

    with r.use_scope_stack(run.stack):
        if info.attempt > 1:  # the plugin's model Activity is retried by Temporal after a Worker kill
            r.scope.event("temporal_retry", handle=run.handle, data={"activity": info.activity_type, "attempt": info.attempt})
    try:
        with r.use_scope_stack(run.stack):
            summary = await r.llm.execute("nim", r.LLMRequest({"model": model_name, "worker": _Cfg.identity, "attempt": str(info.attempt)}, request), provider,
                                          handle=run.handle, model_name=model_name)
    except BaseException:
        run.llm_calls += 1
        _after(run)
        raise
    run.llm_calls += 1
    if isinstance(summary, dict) and summary.get("fallback"):
        run.fallbacks += 1
    _after(run)
    return holder.get("out")


def mark_here(name: str, data: dict[str, Any] | None = None) -> None:
    """A mark on the run's scope from inside an Activity (for example `model_fallback`)."""
    if not enabled() or not activity.in_activity():
        return
    run = _runs.get(activity.info().workflow_id)
    if not run:
        return
    r = _r()
    with r.use_scope_stack(run.stack):
        r.scope.event(name, handle=run.handle, data=_jsonable(data or {}))


async def close_here(*, reason: str) -> list[str]:
    """Close this Worker's scope for the run and return every relay file of the run, from every process that took part."""
    if not enabled() or not activity.in_activity():
        return []
    key = activity.info().workflow_id
    run = _runs.get(key)
    if run:
        await _close_run(run, reason)
    d = relay_dir(key)
    return sorted(str(p) for p in d.iterdir() if p.is_file() and not p.name.startswith(".")) if d.exists() else []
