"""Play the demo's beats unattended against a running Worker: start a run, approve the search grant, delete the exec
sandbox, SIGKILL the Worker mid model call and bring it back, wait for the review, ask for a revision, approve, bundle.

    uv run rehearsal.py [--no-kill] [--no-delete]
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import uuid

from temporalio.client import WorkflowExecutionStatus

from common import GPU_TASK_QUEUE, TASK_QUEUE, connect
from models import AgentInput, GrantDecision, ReviewDecision
from paths import RECIPE_DIR
from provider import DEFAULT_MODEL
from start_workflow import DEFAULT_BRIEF
from workflows.simulation_agent_workflow import SimulationAgentWorkflow


def _run(*script: str) -> None:
    subprocess.run([sys.executable, str(RECIPE_DIR / script[0]), *script[1:]], cwd=RECIPE_DIR, check=False)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-kill", action="store_true")
    ap.add_argument("--no-delete", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    now = lambda: f"t+{time.time() - t0:5.0f}s"  # noqa: E731
    client = await connect()
    handle = await client.start_workflow(SimulationAgentWorkflow.run, AgentInput(brief=DEFAULT_BRIEF, model=DEFAULT_MODEL, review_timeout_hours=72, gpu_task_queue=GPU_TASK_QUEUE),
                                         id=f"packing-cell-{uuid.uuid4().hex[:8]}", task_queue=TASK_QUEUE)
    print(f"{now()} started {handle.id}", flush=True)

    async def running() -> bool:
        return (await handle.describe()).status == WorkflowExecutionStatus.RUNNING

    async def wait_review(n: int) -> str:
        while await running():
            rid = await handle.query(SimulationAgentWorkflow.pending_review)
            if rid and rid.endswith(f":r{n}"):
                return rid
            pg = await handle.query(SimulationAgentWorkflow.pending_grant)
            if pg:
                await handle.execute_update(SimulationAgentWorkflow.grant, GrantDecision(host=pg["host"], decision="approved", approver="rehearsal", notes=""))
                print(f"{now()} beat 1: grant approved for {pg['host']}", flush=True)
            await asyncio.sleep(2)
        raise SystemExit(f"the run ended before review {n}")

    # beat 1: the operator approves the search host; beats 2 and 3 are played while round 1 is in progress
    for _ in range(300):
        if await handle.query(SimulationAgentWorkflow.pending_grant):
            break
        await asyncio.sleep(1)
    pg = await handle.query(SimulationAgentWorkflow.pending_grant)
    if pg:
        await handle.execute_update(SimulationAgentWorkflow.grant, GrantDecision(host=pg["host"], decision="approved", approver="rehearsal", notes=""))
        print(f"{now()} beat 1: grant approved for {pg['host']}", flush=True)
    if not args.no_delete:  # right after the grant, before the first apply: the apply finds its sandbox gone and the Workflow rebuilds
        print(f"{now()} beat 2: delete the exec sandbox", flush=True)
        _run("delete_sandbox.py", handle.id, "exec")
    if not args.no_kill:
        print(f"{now()} beat 3: SIGKILL the Worker mid model call", flush=True)
        _run("kill_worker.py", handle.id, "--timeout", "300")
    rid = await wait_review(1)
    st = await handle.query(SimulationAgentWorkflow.get_state)
    print(f"{now()} review 1: generation {st.gen}, {len(st.layers)} layers, lane {'passed' if st.lane_checks and st.lane_checks[-1].passed else 'failed or unchecked'}", flush=True)
    await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=rid, action="revise", notes="move the pallet staging 1 m further from lane L2 and keep the conveyor on the west side"))
    print(f"{now()} beat 4: revise", flush=True)
    rid = await wait_review(2)
    st = await handle.query(SimulationAgentWorkflow.get_state)
    print(f"{now()} review 2: generation {st.gen}, {len(st.layers)} layers", flush=True)
    await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=rid, action="approve", notes="approved after the revision"))
    result = await handle.result()
    print(f"{now()} {result.status}: {result.rounds} rounds, generation {result.gen}, layer stack {result.layer_stack_hash[:12]}", flush=True)
    print(f"{now()} beat 5: bundle", flush=True)
    _run("bundle.py", handle.id)


if __name__ == "__main__":
    asyncio.run(main())
