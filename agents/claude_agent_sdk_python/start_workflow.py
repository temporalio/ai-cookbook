"""
Start an Agent Execution Workflow

Submits a prompt to the Claude Agent SDK via Temporal and prints the result.

Usage:
    uv run python -m start_workflow "explain how binary search works"
    uv run python -m start_workflow "what files are in the current directory?"
"""

import asyncio
import sys
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.agent import AgentExecutionWorkflow
from models import AgentInput


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Tell me about recursion"

    input_data = AgentInput(prompt=prompt)

    result = await client.execute_workflow(
        AgentExecutionWorkflow.run,
        input_data,
        id=f"claude-agent-sdk-{uuid.uuid4()}",
        task_queue="claude-agent-sdk-task-queue",
    )

    print(f"\nStatus: {result.status}")
    print(f"Tokens: {result.total_tokens}")
    print(f"Events: {result.num_events}")
    print(f"Time:   {result.processing_time_seconds:.2f}s")
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    asyncio.run(main())
