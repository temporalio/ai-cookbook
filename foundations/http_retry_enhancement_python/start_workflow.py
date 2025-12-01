import asyncio
from pathlib import Path

from temporalio.client import Client
from temporalio.envconfig import ClientConfig

from workflows.hello_world_workflow import HelloWorld
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

    # Submit the Hello World workflow for execution
    result = await client.execute_workflow(
        HelloWorld.run,
        "Tell me about recursion in programming.",
        id="my-workflow-id",
        task_queue="hello-world-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
