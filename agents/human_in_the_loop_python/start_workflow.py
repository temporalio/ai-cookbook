import asyncio
import sys
import uuid

from temporalio.client import Client

from workflows.human_in_the_loop_workflow import HumanInTheLoopWorkflow
from temporalio.contrib.pydantic import pydantic_data_converter
from models.models import WorkflowInput


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    default_request = "Delete all test data from the production database"
    user_request = sys.argv[1] if len(sys.argv) > 1 else default_request
    
    # Use a unique workflow ID so you can run multiple instances
    workflow_id = f"human-in-the-loop-{uuid.uuid4()}"
    
    print(f"Starting workflow with ID: {workflow_id}")
    print(f"User request: {user_request}")
    print("\nWorkflow may pause for approval. Watch the worker output for instructions.\n")

    # Submit the workflow for execution
    workflow_input = WorkflowInput(
        user_request=user_request,
        approval_timeout_seconds=300
    )
    result = await client.execute_workflow(
        HumanInTheLoopWorkflow.run,
        args=[workflow_input],
        id=workflow_id,
        task_queue="human-in-the-loop-task-queue",
    )
    
    print(f"\nWorkflow completed!")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
