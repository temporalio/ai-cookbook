import asyncio

from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.contrib.strands import StrandsPlugin

from workflows.aws_assistant_workflow import AWSAssistantWorkflow

TASK_QUEUE = "strands-aws-assistant-task-queue"


async def main():
    # Match the worker's plugin so the client uses the same data converter.
    client = await Client.connect("localhost:7233", plugins=[StrandsPlugin()])

    print(80 * "-")
    user_input = input("Ask the AWS assistant a question: ")

    result = await client.execute_workflow(
        AWSAssistantWorkflow.run,
        user_input,
        id="strands-aws-assistant",
        task_queue=TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.TERMINATE_EXISTING,
    )

    print(80 * "-")
    print(f"Result: {result}")
    print(80 * "-")


if __name__ == "__main__":
    asyncio.run(main())
