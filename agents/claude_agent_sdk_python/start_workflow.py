"""
Start an Agent Execution Workflow

Submits a prompt to the Claude Agent SDK via Temporal and prints the result.

Usage:
    uv run python -m start_workflow "explain how binary search works"
    uv run python -m start_workflow "what files are in the current directory?"

Resume a previous session:
    uv run python -m start_workflow --resume SESSION_ID "follow up question"
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

    # Parse arguments: --resume SESSION_ID is optional
    args = sys.argv[1:]
    resume_session_id = None

    if len(args) >= 2 and args[0] == "--resume":
        resume_session_id = args[1]
        args = args[2:]

    prompt = args[0] if args else "Tell me about recursion"

    input_data = AgentInput(
        prompt=prompt,
        resume_session_id=resume_session_id,
    )

    result = await client.execute_workflow(
        AgentExecutionWorkflow.run,
        input_data,
        id=f"claude-agent-sdk-{uuid.uuid4()}",
        task_queue="claude-agent-sdk-task-queue",
    )

    print(f"\nStatus:     {result.status}")
    print(f"Tokens:     {result.total_tokens}")
    print(f"Events:     {result.num_events}")
    print(f"Time:       {result.processing_time_seconds:.2f}s")
    if result.session_id:
        print(f"Session ID: {result.session_id}")
    print(f"\nResponse:\n{result.response}")

    # Print resume hint
    if result.session_id:
        print(f"\n--- To resume this session: ---")
        print(f'uv run python -m start_workflow --resume {result.session_id} "your follow-up"')


if __name__ == "__main__":
    asyncio.run(main())
