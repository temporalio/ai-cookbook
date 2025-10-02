import asyncio

from temporalio.client import Client

from temporalio.common import WorkflowIDReusePolicy
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin
from workflows.hello_world_workflow import HelloWorldAgent

async def main():
    client = await Client.connect(
        "localhost:7233",
        # Use the plugin to configure Temporal for use with OpenAI Agents SDK
        plugins=[OpenAIAgentsPlugin()],
    )

    # Get user input
    user_input = input("Enter a question: ")

    # Submit the Hello World Agent workflow for execution
    result = await client.execute_workflow(
        HelloWorldAgent.run,
        user_input,
        id="my-workflow-id",
        task_queue="hello-world-openai-agent-task-queue",
        id_reuse_policy=WorkflowIDReusePolicy.TERMINATE_IF_RUNNING,
    )
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())