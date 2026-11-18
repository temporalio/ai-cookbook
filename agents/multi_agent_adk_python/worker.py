import asyncio
import logging

# ADK 2.x imports its tracing module lazily on the first agent run. If that
# first import happens inside the Workflow sandbox, the module's
# `time.monotonic()` call is restricted and the first Workflow Task fails
# (it then succeeds on retry). Importing it here, outside the sandbox, avoids
# that failed task.
import google.adk.telemetry.node_tracing  # noqa: F401
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.envconfig import ClientConfig
from temporalio.google_adk import GoogleAdkPlugin
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

    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(
        **config,
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
