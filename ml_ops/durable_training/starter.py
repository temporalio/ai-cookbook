"""CLI entrypoint for checkpoint-aware BERT training.

This script drives the ``CheckpointedBertTrainingWorkflow`` in the
``bert_checkpointing`` package, which:

- Creates (or reuses) a dataset snapshot for reproducible training.
- Runs checkpoint-aware fine-tuning that can resume from a prior checkpoint.
"""

import asyncio
import random

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.train_tune.bert_checkpointing.custom_types import (
    BertFineTuneConfig,
    BertInferenceRequest,
)
from src.workflows.train_tune.bert_checkpointing.workflow import (
    BertInferenceWorkflow,
    CheckpointedBertTrainingWorkflow,
)


async def main() -> None:
    """Execute a sample checkpoint-aware BERT training workflow."""
    # 1. Connect to the Temporal server using the Pydantic data converter so
    #    that our Pydantic models can be sent over the wire transparently.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Describe a single training configuration. The workflow will:
    #    - Snapshot the dataset for this configuration.
    #    - Run checkpoint-aware fine-tuning over that snapshot.
    config: BertFineTuneConfig = BertFineTuneConfig(
        model_name="bert-base-uncased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=16,
        learning_rate=2e-5,
        max_seq_length=128,
        use_gpu=True,
        seed=random.randint(0, 10000),
        # Use a stable run_id so the checkpoint directory lines up with
        # the default inference script configuration.
        run_id="bert-checkpointed-training-demo-id",
    )

    # 3. Start the checkpointed training workflow and wait for the result.
    training_result = await client.execute_workflow(
        CheckpointedBertTrainingWorkflow.run,
        config,
        id="bert-checkpointed-training-demo-id",
        task_queue="bert-checkpointing-task-queue",
    )

    # 4. Print a concise summary of the run, including how many checkpoints were saved along the way.
    print("\nCheckpointed BERT training result:")
    print(
        f"- run_id={training_result.run_id}, "
        f"epochs={training_result.config.num_epochs}, "
        f"batch_size={training_result.config.batch_size}, "
        f"train_loss={training_result.train_loss:.4f}, "
        f"eval_acc={training_result.eval_accuracy if training_result.eval_accuracy is not None else 'N/A'}, "
        f"time_s={training_result.training_time_seconds:.1f}, "
        f"checkpoints_saved={training_result.total_checkpoints_saved}",
    )
    print("\nStarting inference...")

    # 5. Point at a specific fine-tuned run and provide a batch of texts to score.
    #    Typically this run_id will match the workflow ID used for training.
    request = BertInferenceRequest(
        # Use the same run_id that the training workflow used to name
        # the checkpoint directory under ./bert_runs.
        run_id=training_result.run_id,
        texts=[
            "This movie was great!",
            "I thought it was just okay.",
            "This was a terrible experience.",
            "I didn't see the movie!",
            "The movie is awfully badass.",
        ],
        max_seq_length=128,
        use_gpu=True,
    )

    # 6. Execute the inference workflow and wait for the aggregated result.
    inference_result = await client.execute_workflow(
        BertInferenceWorkflow.run,
        request,
        id="bert-checkpointed-inference-demo-id",
        task_queue="bert-checkpointing-task-queue",
    )

    # 7. Print a compact report for each input text.
    for text, label, score in zip(
        inference_result.texts,
        inference_result.predicted_labels,
        inference_result.confidences,
        strict=True,
    ):
        print(f"{text!r} -> label={label}, confidence={score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
