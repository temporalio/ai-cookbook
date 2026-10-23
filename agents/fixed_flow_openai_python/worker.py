import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities.run_agent_stage import (
    analyze_application,
    critique_analysis,
    draft_decision,
)
from workflows.fixed_flow_workflow import FixedFlowWorkflow

TASK_QUEUE = "fixed-flow-openai-task-queue"


async def main() -> None:
    client = await Client.connect(
        "localhost:7233", data_converter=pydantic_data_converter
    )
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FixedFlowWorkflow],
        activities=[analyze_application, critique_analysis, draft_decision],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
