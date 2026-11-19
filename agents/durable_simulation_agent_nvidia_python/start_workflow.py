"""Start a run.

    uv run start_workflow.py ["brief"]
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from common import GPU_TASK_QUEUE, TASK_QUEUE, connect
from models import AgentInput
from provider import DEFAULT_MODEL
from workflows.simulation_agent_workflow import SimulationAgentWorkflow

DEFAULT_BRIEF = ("Peak-season re-layout. Add packing cell 2: two packing stations, a short conveyor run, pallet staging, floor markings and a "
                 "task light, within 10 m of Dock 4, so an AMR can service it, without encroaching on forklift lane L2 (keep >= 0.6 m clearance "
                 "from every lane) or the painted keep-clear zone. Catalog assets only.")


async def main() -> None:
    brief = " ".join(sys.argv[1:]) or DEFAULT_BRIEF
    client = await connect()
    handle = await client.start_workflow(SimulationAgentWorkflow.run, AgentInput(brief=brief, model=DEFAULT_MODEL, review_timeout_hours=72, gpu_task_queue=GPU_TASK_QUEUE),
                                         id=f"packing-cell-{uuid.uuid4().hex[:8]}", task_queue=TASK_QUEUE)
    print(handle.id)
    print(f"watch it: temporal workflow show -w {handle.id}   (or the Temporal UI)")
    print(f"operator: uv run grant.py {handle.id} approved   |   uv run review.py {handle.id} approve")


if __name__ == "__main__":
    asyncio.run(main())
