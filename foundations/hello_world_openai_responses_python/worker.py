import asyncio
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.envconfig import ClientConfig

from workflows.hello_world_workflow import HelloWorld
from activities import openai_responses
from temporalio.contrib.pydantic import pydantic_data_converter


async def main():
    config_dir = Path(__file__).parent.parent.parent
    config_file = config_dir / "config.toml"
    if not config_file.exists():
        config_file = config_dir / "config.toml.example"
    connect_config = ClientConfig.load_client_connect_config(
        config_file=str(config_file)
    )
    client = await Client.connect(
        **connect_config,
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="hello-world-python-task-queue",
        workflows=[
            HelloWorld,
        ],
        activities=[
            openai_responses.create,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
