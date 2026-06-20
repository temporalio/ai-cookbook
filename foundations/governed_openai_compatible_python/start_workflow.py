import asyncio
import os
import uuid

from temporalio.client import Client

from workflows.governed_model_workflow import GovernedModelWorkflow


async def main():
    client = await Client.connect("localhost:7233")
    workflow_id = f"governed-model-{uuid.uuid4().hex[:12]}"
    result = await client.execute_workflow(
        GovernedModelWorkflow.run,
        {
            "prompt": "Explain why durable retries belong to the workflow runtime.",
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        },
        id=workflow_id,
        task_queue="governed-openai-compatible-task-queue",
    )
    print(f"Workflow ID: {workflow_id}")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
