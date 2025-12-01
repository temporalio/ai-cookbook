import asyncio
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.envconfig import ClientConfig

from workflows.get_weather_workflow import ToolCallingWorkflow
from activities import openai_responses, get_weather_alerts
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
        task_queue="tool-calling-python-task-queue",
        workflows=[
            ToolCallingWorkflow,
        ],
        activities=[
            openai_responses.create,
            get_weather_alerts.get_weather_alerts,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
