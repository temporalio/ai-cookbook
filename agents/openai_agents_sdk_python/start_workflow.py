import asyncio

from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin
from temporalio.envconfig import ClientConfig

from workflows.hello_world_workflow import HelloWorldAgent


async def main():
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(
        **config,
        # Use the plugin to configure Temporal for use with OpenAI Agents SDK
        plugins=[OpenAIAgentsPlugin()],
    )

    # Start workflow
    print( 80 * "-" )

    # Get user input
    user_input = input("Enter a question: ")

    # Submit the Hello World Agent workflow for execution
    result = await client.execute_workflow(
        HelloWorldAgent.run,
        user_input,
        id="my-workflow-id",
        task_queue="hello-world-openai-agent-task-queue",
        id_conflict_policy=WorkflowIDConflictPolicy.TERMINATE_EXISTING,
    )
    print(f"Result: {result}")

    # End of workflow
    print( 80 * "-" )
    print("Workflow completed")

if __name__ == "__main__":
    asyncio.run(main())