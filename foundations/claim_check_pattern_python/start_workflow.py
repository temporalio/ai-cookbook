import asyncio
import time

from temporalio.client import Client

from claim_check_plugin import ClaimCheckPlugin
from workflows.large_data_workflow import LargeDataProcessingWorkflow
from activities.large_data_processing import LargeDataset, SummaryResult


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
    result: SummaryResult = await client.execute_workflow(
        LargeDataProcessingWorkflow.run,
        large_dataset,  # Pass the large dataset as input
        id=f"claim-check-test-{int(time.time())}",
        task_queue="claim-check-pattern-task-queue",
    )
    
    print("=== CLAIM CHECK PATTERN DEMONSTRATION ===")
    print(f"Workflow completed successfully!")
    print(f"Total items processed: {result.total_items}")
    print(f"Items successfully transformed: {result.processed_items}")
    print(f"Transformation errors: {result.errors}")
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Value Statistics: {result.summary_stats['value_stats']}")
    print(f"Text Statistics: {result.summary_stats['text_stats']}")
    print(f"Score Statistics: {result.summary_stats['score_stats']}")
    print("\n=== TRANSFORMATION DETAILS ===")
    print(f"Transformations applied: {result.transformation_stats['transformations_applied']}")
    print(f"Processing completed at: {result.transformation_stats.get('processed_at', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
