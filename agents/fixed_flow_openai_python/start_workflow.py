import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from models.flow import FixedFlowInput
from worker import TASK_QUEUE
from workflows.fixed_flow_workflow import FixedFlowWorkflow


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )
    result = await client.execute_workflow(
        FixedFlowWorkflow.run,
        FixedFlowInput(
            topic="Should a neighborhood library extend its weekend hours?",
            source_material=(
                "Weekend visits increased 28% this year. A six-month pilot would cost "
                "$18,000. The latest survey had 412 responses, with 71% supporting "
                "longer Saturday hours. No Sunday staffing plan has been proposed."
            ),
        ),
        id="fixed-flow-openai-example",
        task_queue=TASK_QUEUE,
    )

    print("\nAnalysis\n--------")
    print(result.analysis)
    print("\nCritique\n--------")
    print(result.critique)
    print("\nRecommendation\n--------------")
    print(result.recommendation)


if __name__ == "__main__":
    asyncio.run(main())
