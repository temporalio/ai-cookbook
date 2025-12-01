import asyncio
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.envconfig import ClientConfig
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.clean_data_workflow import CleanDataWorkflow
from activities import invoke_model


async def main():
    # Connect to Temporal server with pydantic data converter for our data classes
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

    # Create worker with all workflows and activities
    worker = Worker(
        client,
        task_queue="clean-data-task-queue",
        workflows=[
            CleanDataWorkflow,
        ],
        activities=[
            invoke_model.invoke_model,
        ],
    )

    print("Starting Clean Data Worker...")

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
