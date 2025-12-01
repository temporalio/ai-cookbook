import asyncio
import sys
from pathlib import Path

from temporalio.client import Client
from temporalio.envconfig import ClientConfig

from workflows.get_weather_workflow import ToolCallingWorkflow
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

    query = sys.argv[1] if len(sys.argv) > 1 else "Hello, how are you?"

    # Submit the Tool Calling workflow for execution
    result = await client.execute_workflow(
        ToolCallingWorkflow.run,
        query,
        id="my-workflow-id",
        task_queue="tool-calling-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
