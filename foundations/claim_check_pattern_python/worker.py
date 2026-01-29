import asyncio
import logging
import os
from temporalio.client import Client
from temporalio.worker import Worker

from codec.plugin import ClaimCheckPlugin
from workflows.ai_rag_workflow import AiRagWorkflow
from activities.ai_claim_check import ingest_document, rag_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run the Temporal worker with claim check plugin."""
    # Optionally enable claim check via env (default: enabled)
    claim_check_enabled = os.getenv("CLAIM_CHECK_ENABLED", "true").lower() != "false"
    plugins = [ClaimCheckPlugin()] if claim_check_enabled else []

    # Connect to Temporal server
    client = await Client.connect(
        "localhost:7233",
        plugins=plugins,
    )

    # Create worker (data converter comes from client)
    worker = Worker(
        client,
        task_queue="claim-check-pattern-task-queue",
        workflows=[AiRagWorkflow],
        activities=[ingest_document, rag_answer],
    )

    if claim_check_enabled:
        logger.info("Starting worker with claim check enabled (S3-backed)")
        logger.info("Worker will handle large payloads using S3 storage")
    else:
        logger.info("Starting worker with claim check DISABLED (for demo)")
        logger.info("Large payloads may exceed Temporal payload limits and fail")

    # Run the worker
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
