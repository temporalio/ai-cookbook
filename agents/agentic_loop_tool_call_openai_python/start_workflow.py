import asyncio
import sys
import uuid

from temporalio.client import Client

from workflows.agent import AgentWorkflow
from temporalio.contrib.pydantic import pydantic_data_converter


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    query = sys.argv[1] if len(sys.argv) > 1 else "Tell me about recursion"

    # Submit the the agent workflow for execution
    result = await client.execute_workflow(
        AgentWorkflow.run,
        query,
        id=f"agentic-loop-id-{uuid.uuid4()}",
        task_queue="tool-invoking-agent-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
