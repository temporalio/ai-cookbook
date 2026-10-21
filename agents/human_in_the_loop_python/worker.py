import asyncio
import logging

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

from activities.execute_action import execute_action
from activities.notify_approval_needed import notify_approval_needed
from activities.openai_responses import create
from workflows.human_in_the_loop_workflow import HumanInTheLoopWorkflow


async def main():
    # Configure logging to see workflow.logger output
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

    worker = Worker(
        client,
        task_queue="human-in-the-loop-task-queue",
        workflows=[HumanInTheLoopWorkflow],
        activities=[
            create,
            execute_action,
            notify_approval_needed,
        ],
    )
    
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
