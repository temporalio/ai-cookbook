"""
Temporal Worker for Claude Agent SDK Example

Starts a worker that listens for AgentExecutionWorkflow tasks and dispatches
them to the registered activities.
"""

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.agent import AgentExecutionWorkflow
from activities.agent_executor import execute_agent_activity, log_result_activity


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="claude-agent-sdk-task-queue",
        workflows=[AgentExecutionWorkflow],
        activities=[execute_agent_activity, log_result_activity],
    )

    print("Worker started, listening on claude-agent-sdk-task-queue")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
