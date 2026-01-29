"""Temporal worker for the BERT checkpointing, training, and inference example."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.train_tune.bert_checkpointing.bert_activities import (
    BertCheckpointingActivities,
    BertFineTuneActivities,
    BertInferenceActivities,
)
from src.workflows.train_tune.bert_checkpointing.workflow import (
    BertInferenceWorkflow,
    CheckpointedBertTrainingWorkflow,
)

# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------


async def main() -> None:
    # 1. Connect to Temporal Server using the same Pydantic data converter
    # used by the starter script so typed models round-trip cleanly.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Set the task queue that this worker will poll. This must match the
    # ``task_queue`` used when starting workflows from ``starter.py``.
    task_queue = "bert-checkpointing-task-queue"

    # 3. Instantiate activity collections.
    fine_tune_activities = BertFineTuneActivities()
    inference_activities = BertInferenceActivities()
    checkpointing_activities = BertCheckpointingActivities()

    # 4. Build worker
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[CheckpointedBertTrainingWorkflow, BertInferenceWorkflow],
        activities=[
            fine_tune_activities.fine_tune_bert,
            inference_activities.run_bert_inference,
            checkpointing_activities.create_dataset_snapshot,
        ],
        activity_executor=ThreadPoolExecutor(5),
    )

    # 5. Run worker
    await worker.run()


# CLI Hook
if __name__ == "__main__":
    asyncio.run(main())
