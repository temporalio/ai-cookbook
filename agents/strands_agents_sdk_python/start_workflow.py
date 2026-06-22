import asyncio
import uuid
from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.agent import StrandsAgentWorkflow


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    print("=" * 80)
    print("Strands Agent Chat (type 'exit' or 'quit' to end)")
    print("=" * 80)

    while True:
        print()
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            result = await client.execute_workflow(
                StrandsAgentWorkflow.run,
                user_input,
                id=f"strands-agent-{uuid.uuid4()}",
                task_queue="strands-agent-task-queue",
                id_reuse_policy=WorkflowIDReusePolicy.TERMINATE_IF_RUNNING,
            )
            print(f"\nAgent: {result}")
            print("-" * 80)

        except Exception as e:
            print(f"\nError: {e}")
            print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())