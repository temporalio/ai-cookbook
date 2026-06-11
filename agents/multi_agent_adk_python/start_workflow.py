import asyncio
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from models.models import AssignmentInput
from workflows.assignment_workflow import (
    TASK_QUEUE,
    MultiAgentAssignmentWorkflow,
)


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    order = AssignmentInput(
        order_id="order-001",
        hotel="Caesars Palace",
        priority="VIP",
        servings=50,
        deadline_minutes=25,
        delivery_lat=36.1162,
        delivery_lng=-115.1745,
    )

    workflow_id = f"multi-agent-adk-{uuid.uuid4()}"
    print(f"Starting workflow {workflow_id}")
    print(f"Order: {order.model_dump()}\n")

    result = await client.execute_workflow(
        MultiAgentAssignmentWorkflow.run,
        order,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    print(f"Assigned driver: {result.driver_id}")
    print(f"Reasoning:       {result.reasoning_summary}")


if __name__ == "__main__":
    asyncio.run(main())
