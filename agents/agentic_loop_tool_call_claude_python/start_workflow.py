import asyncio
import sys
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.envconfig import ClientConfig

from workflows.agent import AgentWorkflow


async def main():
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(
        **config,
        data_converter=pydantic_data_converter,
    )

    query = sys.argv[1] if len(sys.argv) > 1 else "Tell me about recursion"

    # Submit the agent workflow for execution
    result = await client.execute_workflow(
        AgentWorkflow.run,
        query,
        id=f"agentic-loop-claude-id-{uuid.uuid4()}",
        task_queue="tool-invoking-agent-claude-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
