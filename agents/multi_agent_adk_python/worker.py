import asyncio
import logging

from temporalio.client import Client
from temporalio.contrib.google_adk_agents import GoogleAdkPlugin
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities.tools import (
    tool_get_fleet_status,
    tool_get_order_priorities,
    tool_get_route_info,
)
from workflows.assignment_workflow import (
    TASK_QUEUE,
    MultiAgentAssignmentWorkflow,
)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    # GoogleAdkPlugin registers the `invoke_model` activity (used by
    # TemporalModel for LLM calls) and provides the workflow-sandbox
    # passthroughs and deterministic runtime overrides ADK needs.
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiAgentAssignmentWorkflow],
        activities=[
            tool_get_fleet_status,
            tool_get_order_priorities,
            tool_get_route_info,
        ],
        plugins=[GoogleAdkPlugin()],
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
