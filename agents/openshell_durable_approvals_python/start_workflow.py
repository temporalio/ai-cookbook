"""Start a run.

    uv run python -m start_workflow ["brief"]
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from common import TASK_QUEUE, connect
from models import AgentInput
from provider import DEFAULT_MODEL
from workflows.approval_agent_workflow import ApprovalAgentWorkflow

DEFAULT_BRIEF = ("Lay out a small packing cell: two packing tables, one short conveyor and two pallets, at least 0.6 m apart, "
                 "in a 6 m by 4 m area. Write the placements to layout.json.")


async def main() -> None:
    brief = " ".join(sys.argv[1:]) or DEFAULT_BRIEF
    client = await connect()
    handle = await client.start_workflow(ApprovalAgentWorkflow.run, AgentInput(brief=brief, model=DEFAULT_MODEL, review_timeout_hours=72),
                                         id=f"packing-cell-{uuid.uuid4().hex[:8]}", task_queue=TASK_QUEUE)
    print(handle.id)
    print(f"watch it: temporal workflow show -w {handle.id}   (or the Temporal UI)")
    print(f"operator: uv run python -m grant {handle.id} approved   ·   uv run python -m review {handle.id} approve")


if __name__ == "__main__":
    asyncio.run(main())
