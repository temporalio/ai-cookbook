import asyncio
import uuid

from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin
from workflows.streaming_agent_workflow import StreamingAgent


async def main():
    client = await Client.connect(
        "localhost:7233",
        # Use the plugin to configure Temporal for use with OpenAI Agents SDK
        plugins=[OpenAIAgentsPlugin()],
    )

    print(80 * "-")

    # Get user input
    user_input = input("Enter a question: ")

    workflow_id = f"streaming-agent-{uuid.uuid4()}"

    # Start the workflow and return immediately -- we don't block on the
    # result here, since the point of this recipe is to observe the run's
    # progress concurrently via subscribe.py.
    await client.start_workflow(
        StreamingAgent.run,
        user_input,
        id=workflow_id,
        task_queue="streaming-openai-agent-task-queue",
        id_conflict_policy=WorkflowIDConflictPolicy.FAIL,
    )

    print(f"Started workflow: {workflow_id}")
    print(80 * "-")
    print("In another terminal, run:")
    print(f"  uv run python -m subscribe {workflow_id}")


if __name__ == "__main__":
    asyncio.run(main())
