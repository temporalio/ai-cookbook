import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from activities.openai_compatible import invoke_model
from workflows.governed_model_workflow import GovernedModelWorkflow


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="governed-openai-compatible-task-queue",
        workflows=[GovernedModelWorkflow],
        activities=[invoke_model],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
