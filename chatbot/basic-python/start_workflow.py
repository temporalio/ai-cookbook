import asyncio

from temporalio.client import Client

from workflows.hello_world_workflow import HelloWorld


async def main():
    client = await Client.connect(
        "localhost:7233",
    )

    # Submit the Hello World workflow for execution
    result = await client.execute_workflow(
        HelloWorld.run,
        "Tell me about recursion in programming.",
        id="my-workflow-id",
        task_queue="hello-world-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
