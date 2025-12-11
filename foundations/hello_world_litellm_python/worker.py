import asyncio

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.envconfig import ClientConfig

from workflows.hello_world_workflow import HelloWorld
from activities import litellm_completion

from temporalio.contrib.pydantic import pydantic_data_converter


async def main():
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(
        **config,
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="hello-world-python-task-queue",
        workflows=[
            HelloWorld,
        ],
        activities=[
            litellm_completion.create,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
