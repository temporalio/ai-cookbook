import asyncio
from pathlib import Path

from temporalio.client import Client
from temporalio.envconfig import ClientConfig

from temporalio.common import WorkflowIDReusePolicy
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin
from workflows.hello_world_workflow import HelloWorldAgent

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