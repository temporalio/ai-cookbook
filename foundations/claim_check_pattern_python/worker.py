import asyncio
import logging
from temporalio.client import Client
from temporalio.worker import Worker

from claim_check_plugin import ClaimCheckPlugin
from workflows.large_data_workflow import LargeDataProcessingWorkflow
from activities.large_data_processing import process_large_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run the Temporal worker with claim check plugin."""
    # Connect to Temporal server with claim check plugin
    client = await Client.connect(
        "localhost:7233",
        plugins=[ClaimCheckPlugin()]
    )
    
    # Create worker (data converter comes from client)
    worker = Worker(
        client,
        task_queue="claim-check-pattern-task-queue",
        workflows=[LargeDataProcessingWorkflow],
        activities=[process_large_dataset],
    )
    
    logger.info("Starting worker with claim check plugin...")
    logger.info("Worker will handle large payloads using Redis storage")
    
    # Run the worker
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
