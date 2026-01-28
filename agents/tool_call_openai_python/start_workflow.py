import asyncio
import sys

from temporalio.client import Client
from temporalio.envconfig import ClientConfig

from workflows.get_weather_workflow import ToolCallingWorkflow
from temporalio.contrib.pydantic import pydantic_data_converter


async def main():
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(
        **config,
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
