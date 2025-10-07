import asyncio
import time

from temporalio.client import Client

from claim_check_plugin import ClaimCheckPlugin
from workflows.large_data_workflow import LargeDataProcessingWorkflow
from activities.large_data_processing import LargeDataset


async def main():
    client = await Client.connect(
        "localhost:7233",
        plugins=[ClaimCheckPlugin()]
    )

    # Create a large dataset to pass to the workflow
    large_dataset = LargeDataset(
        data=[
            {
                "id": f"item_{i}",
                "value": i * 10,
                "text": f"This is sample text for item {i}. " * 10,  # Make it larger
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "category": f"category_{i % 10}",
                    "tags": [f"tag_{j}" for j in range(i % 5)]
                }
            }
            for i in range(2000)  # Create 2000 items for a large payload
        ],
        metadata={
            "generated_at": "2024-01-01T00:00:00Z",
            "total_items": 2000,
            "description": "Sample large dataset for claim check pattern demonstration"
        }
    )

    # Submit the Large Data Processing workflow for execution
    result = await client.execute_workflow(
        LargeDataProcessingWorkflow.run,
        large_dataset,  # Pass the large dataset as input
        id=f"claim-check-test-{int(time.time())}",
        task_queue="claim-check-pattern-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
