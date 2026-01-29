"""Checkpoint-aware Temporal workflows for BERT training and inference.

This module demonstrates how to:

- Create a reproducible dataset snapshot once and reuse it across runs.
- Run checkpoint-aware fine-tuning activities that can resume from a prior
  checkpoint path.
- Track the latest checkpoint in workflow state via signals so that external
  clients can orchestrate resumptions.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.workflows.train_tune.bert_checkpointing.custom_types import (
        BertFineTuneConfig,
        BertFineTuneRequest,
        BertFineTuneResult,
        BertInferenceRequest,
        BertInferenceResult,
        CheckpointInfo,
        DatasetSnapshotRequest,
        DatasetSnapshotResult,
    )


# ----------------------------------------------------------------------------------
# Checkpointed BERT Training Workflow
# ----------------------------------------------------------------------------------
@workflow.defn
class CheckpointedBertTrainingWorkflow:
    """Workflow that runs checkpoint-aware fine-tuning with a shared snapshot.

    The pattern here is:

    1. Materialize (or reuse) a dataset snapshot so that training becomes
       reproducible across workers and retries.
    2. Run a single long-lived training activity that periodically saves model
       checkpoints and reports progress through signals.
    3. Expose lightweight queries so external clients can inspect the most
       recent checkpoint while the run is still in flight.
    """

    def __init__(self) -> None:
        self.latest_checkpoint: CheckpointInfo | None = None
        self.run_id = None

    @workflow.signal
    def update_checkpoint(self, info: CheckpointInfo) -> None:
        """Record the most recent checkpoint information in workflow state.

        Activities call this signal whenever a new checkpoint is persisted so
        that the workflow can expose it via :py:meth:`get_latest_checkpoint`.
        """
        self.latest_checkpoint = info

    @workflow.query
    def get_latest_checkpoint(self) -> CheckpointInfo | None:
        """Expose the most recently recorded checkpoint (if any)."""
        return self.latest_checkpoint

    # Main Workflow Function
    @workflow.run
    async def run(self, config: BertFineTuneConfig) -> BertFineTuneResult:
        """Run a single checkpoint-aware fine-tuning job."""
        # Prefer a coordinator-provided run_id when present so that the
        # coordinator can control how training/eval artifacts are named and the
        # evaluation workflow can later find the right checkpoint directory.
        if config.run_id:
            run_id = config.run_id
        else:
            # Fallback for direct usage without a coordinator: derive a
            # human-friendly, unique run identifier from Temporal's run_id.
            run_id = f"bert-checkpointed-{workflow.info().run_id}"
            config.run_id = run_id

        self.run_id = run_id

        workflow.logger.info(
            "Starting checkpointed BERT run for model %s on %s/%s",
            config.model_name,
            config.dataset_name,
            config.dataset_config_name,
        )

        # Step 1: Materialize (or reuse) a dataset snapshot for this configuration.
        snapshot_request = DatasetSnapshotRequest(
            run_id=run_id,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config_name,
            max_samples=config.max_train_samples,
        )

        snapshot: DatasetSnapshotResult = await workflow.execute_activity(
            "create_dataset_snapshot",
            snapshot_request,
            start_to_close_timeout=timedelta(minutes=10),
        )

        # Step 2: Run checkpoint-aware fine-tuning, optionally resuming from the
        # latest known checkpoint path (if one has already been recorded).
        resume_from = self.latest_checkpoint.path if self.latest_checkpoint else None

        request = BertFineTuneRequest(
            run_id=run_id,
            config=config,
            dataset_snapshot=snapshot,
            resume_from_checkpoint=resume_from,
        )

        result: BertFineTuneResult | dict = await workflow.execute_activity(
            "fine_tune_bert",
            request,
            start_to_close_timeout=timedelta(hours=2),
        )

        # Guard against cases where the Pydantic data converter returns a plain
        # dict instead of a model instance (e.g., if imports are misaligned).

        if isinstance(result, dict):
            run_id = result.get("run_id")
            checkpoints_saved = result.get("total_checkpoints_saved")

        else:
            run_id = result.run_id
            checkpoints_saved = result.total_checkpoints_saved

        workflow.logger.info(
            "Completed checkpointed BERT run %s (checkpoints_saved=%s)",
            run_id,
            checkpoints_saved,
        )

        # If we got a dict back, re-wrap it as a BertFineTuneResult so callers
        # see a consistent type.
        if isinstance(result, dict):
            return BertFineTuneResult(**result)
        return result


@workflow.defn
class BertInferenceWorkflow:
    """Workflow that runs inference using a fine-tuned BERT checkpoint."""

    @workflow.run
    async def run(self, input: BertInferenceRequest) -> BertInferenceResult:
        """Execute BERT inference for a batch of texts."""
        # Handle both model instances and plain dicts defensively.
        if isinstance(input, dict):
            run_id = input.get("run_id")
            texts = input.get("texts", [])
        else:
            run_id = input.run_id
            texts = input.texts

        workflow.logger.info(
            "Starting BERT inference workflow for run %s on %s text(s)",
            run_id,
            len(texts),
        )
        result: BertInferenceResult | dict = await workflow.execute_activity(
            "run_bert_inference",
            input,
            start_to_close_timeout=timedelta(minutes=10),
        )
        if isinstance(result, dict):
            out = BertInferenceResult(**result)
        else:
            out = result
        workflow.logger.info("Completed BERT inference workflow for run %s", run_id)
        return out
