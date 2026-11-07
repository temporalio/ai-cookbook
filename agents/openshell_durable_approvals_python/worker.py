"""The worker: the workflow, the sandbox activities, and the model activity the Agents plugin registers.

    uv run python -m worker
"""

from __future__ import annotations

import asyncio

from temporalio.worker import Worker

from activities import sandbox as sb
from activities.openshell import backend
from common import TASK_QUEUE, connect
from workflows.approval_agent_workflow import ApprovalAgentWorkflow


async def main() -> None:
    client = await connect(register_model_activities=True)
    worker = Worker(client, task_queue=TASK_QUEUE, workflows=[ApprovalAgentWorkflow],
                    activities=[sb.ensure_sandboxes, sb.release_sandboxes, sb.sandbox_search, sb.sandbox_run_code, sb.apply_grant])
    print(f"worker on task queue {TASK_QUEUE!r}; sandbox backend: {type(backend()).__name__}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
