import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from workflows.human_in_the_loop_workflow import HumanInTheLoopWorkflow
from activities.bedrock import create
from activities.execute_action import execute_action
from activities.notify_approval_needed import notify_approval_needed
from temporalio.contrib.pydantic import pydantic_data_converter
import server
from health import HealthTracker, HealthInterceptor, create_tuner


async def main():
    # Configure logging to see workflow.logger output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    health = HealthTracker()
    worker = Worker(
        client,
        task_queue="human-in-the-loop-task-queue",
        workflows=[HumanInTheLoopWorkflow],
        activities=[
            create,
            execute_action,
            notify_approval_needed,
        ],
        interceptors=[HealthInterceptor(health)],
        tuner=create_tuner(health),
    )

    await asyncio.gather(server.run(health), worker.run())


if __name__ == "__main__":
    asyncio.run(main())
