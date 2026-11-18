"""Chaos: SIGKILL the Worker while a model call is in flight, wait, start it again, and show that Temporal retried only
that Activity. The Worker's pid comes from runs/_worker.json, which worker.py writes.

    uv run kill_worker.py <workflow-id> [--activity invoke_model_activity] [--downtime 8]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time

from temporalio.api.enums.v1 import EventType, PendingActivityState
from temporalio.client import WorkflowExecutionStatus

from common import connect
from paths import RECIPE_DIR, runs_dir


def _in_flight(desc, activity_type: str):
    return [p for p in desc.raw_description.pending_activities
            if p.activity_type.name == activity_type and p.state == PendingActivityState.PENDING_ACTIVITY_STATE_STARTED]


def _tree(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        todo += [int(k) for k in kids]
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow_id")
    ap.add_argument("--activity", default="invoke_model_activity", help="the Activity type to catch in flight")
    ap.add_argument("--downtime", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    info = json.loads((runs_dir() / "_worker.json").read_text())
    client = await connect()
    handle = client.get_workflow_handle(args.workflow_id)
    t0 = time.time()
    print(f"[1] watching {args.workflow_id} for an in-flight {args.activity} ...", flush=True)
    while True:
        desc = await handle.describe()
        if desc.status != WorkflowExecutionStatus.RUNNING:
            raise SystemExit(f"workflow is {desc.status.name}; start a fresh run")
        hits = _in_flight(desc, args.activity)
        if hits:
            for p in reversed(_tree(info["pid"])):  # children first
                with contextlib.suppress(ProcessLookupError):
                    os.kill(p, signal.SIGKILL)
            print(f"[2] SIGKILL worker (pid {info['pid']}) at t+{time.time() - t0:.1f}s: {args.activity} attempt {hits[0].attempt} was mid-flight", flush=True)
            break
        if time.time() - t0 > args.timeout:
            raise SystemExit("timed out waiting for the activity")
        await asyncio.sleep(0.3)
    print(f"[3] no worker alive for {args.downtime:.0f}s; the run is frozen, nothing lost", flush=True)
    await asyncio.sleep(args.downtime)
    log = (runs_dir() / "_worker.log").open("a")
    proc = subprocess.Popen([sys.executable, str(RECIPE_DIR / "worker.py"), *info["argv"]], cwd=RECIPE_DIR, stdout=log, stderr=subprocess.STDOUT)
    print(f"[4] worker back (pid {proc.pid}); waiting for attempt 2 ...", flush=True)
    seen = None
    while time.time() - t0 < args.timeout:
        desc = await handle.describe()
        for p in desc.raw_description.pending_activities:
            if p.activity_type.name == args.activity and p.attempt >= 2:
                seen = (p.attempt, p.last_failure.message if p.last_failure else None)
        if desc.status != WorkflowExecutionStatus.RUNNING or (seen and not _in_flight(desc, args.activity)):
            break
        await asyncio.sleep(0.5)
    print(f"[5] retry observed: attempt={seen[0] if seen else '?'} last_failure={seen[1] if seen else '?'} at t+{time.time() - t0:.1f}s", flush=True)
    for _ in range(120):  # the started event reaches the history when the retried attempt finishes
        desc = await handle.describe()
        if desc.status != WorkflowExecutionStatus.RUNNING or not any(p.activity_type.name == args.activity for p in desc.raw_description.pending_activities):
            break
        await asyncio.sleep(0.5)
    hist = await handle.fetch_history()
    sched, started = {}, {}
    for e in hist.events:
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            sched[e.event_id] = e.activity_task_scheduled_event_attributes.activity_type.name
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            a = e.activity_task_started_event_attributes
            started[a.scheduled_event_id] = (a.attempt, a.last_failure.message if a.HasField("last_failure") else "")
    print("--- activities with attempt > 1 (history) ---")
    for sid, name in sched.items():
        att = started.get(sid)
        if att and att[0] > 1:
            print(f"   {name:28s} attempt {att[0]}  last failure: {att[1][:80]}")
    print("the worker keeps running in the background; stop it with: kill", proc.pid)


if __name__ == "__main__":
    asyncio.run(main())
