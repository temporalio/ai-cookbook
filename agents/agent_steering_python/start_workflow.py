import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.agent_steering_workflow import SteerableAgentWorkflow


async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    handle = await client.start_workflow(
        SteerableAgentWorkflow.run,
        "Draft a plan to migrate our service to a new datacenter.",
        id="agent-steering-example",
        task_queue="agent-steering-task-queue",
    )

    # Nudge the running agent mid-task. The loop folds this into the next model call.
    await handle.signal(SteerableAgentWorkflow.steer, "Prefer the lowest-downtime approach.")

    result = await handle.result()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
