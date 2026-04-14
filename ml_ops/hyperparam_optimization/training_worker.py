"""Temporal worker for BERT training tasks.

This worker is separated from the evaluation/sweep worker so you can:

- Point it at GPU- or accelerator-enabled machines.
- Scale the training capacity independently of the orchestration layer.
- Keep long-running, memory-heavy activities isolated on their own queue.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.train_tune.bert_sweeps.bert_activities import (
    BertCheckpointingActivities,
    BertFineTuneActivities,
)
from src.workflows.train_tune.bert_sweeps.workflows import (
    CheckpointedBertTrainingWorkflow,
)


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
async def main() -> None:
    # 1. Connect to Temporal Server. Use the Pydantic data converter so that
    # our request/response models can be used directly as activity arguments.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Set task queue dedicated to training workloads.
    task_queue = "bert-training-task-queue"

    # 3. Instantiate activity classes that own all ML-specific logic.
    fine_tune_activities = BertFineTuneActivities()
    checkpointing_activities = BertCheckpointingActivities()

    # 4. Build worker
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[CheckpointedBertTrainingWorkflow],
        activities=[
            fine_tune_activities.fine_tune_bert,
            checkpointing_activities.create_dataset_snapshot,
        ],
        activity_executor=ThreadPoolExecutor(5),
        max_concurrent_activities=1,  # Keep max concurrent activities at 1 for local execution to prevent OOM issues
        max_cached_workflows=10,  # (Optional) Keep max cached workflows limited to limit memory consumption
    )

    # 5. Run Worker
    await worker.run()


# CLI hook
if __name__ == "__main__":
    asyncio.run(main())
