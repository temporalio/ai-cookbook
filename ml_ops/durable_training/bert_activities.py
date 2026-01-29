"""Temporal activities for BERT checkpointing, fine-tuning, and inference.

This module contains the *non-deterministic* parts of the BERT checkpointing
example:

- Creating content-addressed dataset snapshots for reproducible experiments.
- Long-running, compute-heavy fine-tuning with mid-run checkpoints.
- Loading a saved checkpoint and running batch inference.

The corresponding Temporal workflows orchestrate these activities, but all of
the actual ML logic (dataset loading, tokenization, model forward passes, etc.)
stays here so that workflow code can remain deterministic and replay-safe.
"""

import asyncio
import contextlib
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Final

from temporalio import activity

from src.workflows.train_tune.bert_checkpointing.custom_types import (
    BertFineTuneRequest,
    BertFineTuneResult,
    BertInferenceRequest,
    BertInferenceResult,
    DatasetSnapshotRequest,
    DatasetSnapshotResult,
)
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Human-friendly error message surfaced when ML dependencies are missing. This keeps
# the Temporal worker process healthy even if the Python environment is not configured
# for running the BERT example.
TRANSFORMERS_IMPORT_MESSAGE: Final[str] = (
    "BERT checkpointing dependencies are not installed. "
    "Install 'transformers', 'datasets', and 'torch' to execute this activity."
)

# How frequently the fine-tuning activity should send heartbeats while training is
# running in a background thread. This example uses a modest interval suitable for
# both local development and the 5s per-test timeout configured in pytest.
HEARTBEAT_INTERVAL_SECONDS: Final[float] = 5.0


# -------------------------------------------------------------------------------
# Fine Tuning Activities
# -------------------------------------------------------------------------------
class BertFineTuneActivities:
    """Activity collection for checkpoint-aware BERT fine-tuning."""

    def __init__(self) -> None:
        self.config = None

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = (predictions == labels).astype("float32").mean().item()
        return {"accuracy": accuracy}

    def tokenize_function(self, batch: dict) -> dict:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
        )

    def _fine_tune_bert_sync(self, request: BertFineTuneRequest) -> BertFineTuneResult:
        """Run a BERT fine-tuning job with optional checkpointing.

        This helper encapsulates *all* ML details and knows nothing about Temporal.
        The async activity wrapper offloads to this helper in a thread so that:

        - The code can be imported and unit-tested without a Temporal worker.
        - The Temporal worker can keep polling for new tasks while training runs.
        """
        if torch is None or load_dataset is None:
            # pragma: no cover - only hit when deps are actually missing
            raise RuntimeError(TRANSFORMERS_IMPORT_MESSAGE)

        start_time = time.perf_counter()
        self.config = request.config

        # ------------------------------------------------------------------
        # 1. Choose an appropriate device (CUDA, Apple MPS, or CPU).
        # ------------------------------------------------------------------
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            self.config.use_gpu
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # ------------------------------------------------------------------
        # 2. Load the dataset, preferring a pre-materialized snapshot when
        #    provided for full reproducibility.
        # ------------------------------------------------------------------
        if request.dataset_snapshot is not None:
            snapshot_path = Path(request.dataset_snapshot.snapshot_path)
            data_path = snapshot_path / "data.jsonl"
            raw_datasets = load_dataset("json", data_files=str(data_path))
        else:
            raw_datasets = load_dataset(self.config.dataset_name, self.config.dataset_config_name)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Apply the tokenizer across the dataset; `batched=True` lets HF process
        # multiple rows at once for better throughput.
        tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        # Tell Datasets to yield PyTorch tensors for the columns the Trainer needs.
        tokenized_datasets.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets.get("validation") or tokenized_datasets.get(
            "validation_matched",
        )

        # 3. Optionally sub-sample train/eval for a fast demo run on laptops.
        if self.config.max_train_samples is not None and self.config.max_train_samples < len(
            train_dataset,
        ):
            train_dataset = train_dataset.select(range(self.config.max_train_samples))
        if (
            eval_dataset is not None
            and self.config.max_eval_samples is not None
            and self.config.max_eval_samples < len(eval_dataset)
        ):
            eval_dataset = eval_dataset.select(range(self.config.max_eval_samples))

        # 4. Construct the classification head on top of the base encoder.
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
        )
        model.to(device)

        # 5. Configure the Transformers Trainer with step-based checkpointing.
        #    We keep the configuration simple and tuned for readability over
        #    state-of-the-art results. If there's no eval dataset (e.g. when
        #    training purely on a JSONL snapshot with only a train split),
        #    disable evaluation to avoid Trainer complaining.
        steps_per_epoch = max(1, len(train_dataset) // self.config.batch_size)
        # Aim for a couple of checkpoints per epoch when possible.
        save_steps = max(1, steps_per_epoch // 2)

        training_args = TrainingArguments(
            output_dir=f"./bert_runs/{request.run_id}",
            num_train_epochs=float(self.config.num_epochs),
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            logging_strategy="steps",
            logging_steps=save_steps,
            report_to=[],
            load_best_model_at_end=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics if eval_dataset is not None else None,
        )

        # If we're resuming after a worker restart or a prior run, prefer an
        # explicitly provided checkpoint path, otherwise detect the latest
        # checkpoint in the output directory (if any) so we don't start from 0.
        output_dir = Path(training_args.output_dir)
        resume_path = request.resume_from_checkpoint
        if resume_path is None and output_dir.exists():
            existing_checkpoints = sorted(p for p in output_dir.glob("checkpoint-*") if p.is_dir())
            if existing_checkpoints:
                resume_path = str(existing_checkpoints[-1])

        train_result = trainer.train(resume_from_checkpoint=resume_path)

        metrics: dict = {}
        if eval_dataset is not None:
            metrics = trainer.evaluate()

        # Persist the final fine-tuned model and tokenizer so that the inference
        # activity can load them later based solely on ``run_id``.
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

        # Discover any mid-run checkpoints created by the Trainer.
        checkpoint_dirs = sorted(p for p in output_dir.glob("checkpoint-*") if p.is_dir())
        total_checkpoints_saved = len(checkpoint_dirs)

        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        training_time_seconds = float(time.perf_counter() - start_time)

        return BertFineTuneResult(
            run_id=request.run_id,
            config=self.config,
            train_loss=float(train_result.training_loss),
            eval_accuracy=float(metrics.get("eval_accuracy"))
            if "eval_accuracy" in metrics
            else None,
            training_time_seconds=training_time_seconds,
            num_parameters=num_parameters,
            dataset_snapshot=request.dataset_snapshot,
            total_checkpoints_saved=total_checkpoints_saved,
        )

    @activity.defn
    async def fine_tune_bert(self, request: BertFineTuneRequest) -> BertFineTuneResult:
        """Temporal activity that runs a BERT fine-tuning job."""
        activity.logger.info(
            "Starting BERT fine-tuning run %s with model %s on %s/%s",
            request.run_id,
            request.config.model_name,
            request.config.dataset_name,
            request.config.dataset_config_name,
        )

        # Offload the training to a separate thread and send periodic heartbeats
        # so Temporal can detect liveness during long-running fine-tuning.
        #
        # This pattern lets us:
        # - Keep the heavy ML work off the event loop thread, and
        # - Give Temporal visibility into progress via heartbeats, which in turn
        #   enables heartbeat timeouts and cancellation handling.
        training_task = asyncio.create_task(asyncio.to_thread(self._fine_tune_bert_sync, request))
        try:
            while not training_task.done():
                activity.heartbeat({"run_id": request.run_id})
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            result = await training_task
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            training_task.cancel()
            with contextlib.suppress(Exception):
                await training_task
            raise

        activity.logger.info(
            "Completed BERT fine-tuning run %s with loss %.4f and accuracy %s",
            request.run_id,
            result.train_loss,
            "N/A" if result.eval_accuracy is None else f"{result.eval_accuracy:.3f}",
        )
        return result


# -------------------------------------------------------------------------------
# TEMP: Inference Activity
# -------------------------------------------------------------------------------
class BertInferenceActivities:
    def __init__(self):
        pass

    def _run_bert_inference_sync(self, request: BertInferenceRequest) -> BertInferenceResult:
        """Run batch inference using a fine-tuned BERT model checkpoint.

        As with ``_fine_tune_bert_sync``, this helper is intentionally free of any
        Temporal-specific APIs. It simply loads a saved model and tokenizer from
        disk and runs a forward pass over a batch of texts.
        """
        # Device selection mirroring the training configuration.
        if request.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            request.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Each training run writes its artifacts into a run-specific directory under
        # ``./bert_runs``. The ``run_id`` flowing through the workflow and activity
        # input acts as the glue between training and inference.
        model_dir = f"./bert_runs/{request.run_id}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        # Tokenize the input texts into a padded batch of tensors that the model
        # can consume. We keep the interface high-level on purpose so that callers
        # do not have to manage token IDs directly.
        encoded = tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            max_length=request.max_seq_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Standard PyTorch inference boilerplate: no gradients, softmax over logits,
        # and then take the argmax and associated probability for each example.
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidences, predicted = probs.max(dim=-1)

        predicted_labels = predicted.tolist()
        confidence_scores = confidences.tolist()

        return BertInferenceResult(
            run_id=request.run_id,
            texts=list(request.texts),
            predicted_labels=predicted_labels,
            confidences=confidence_scores,
        )

    @activity.defn
    async def run_bert_inference(self, request: BertInferenceRequest) -> BertInferenceResult:
        """Temporal activity that runs BERT inference using a fine-tuned checkpoint."""
        activity.logger.info(
            "Starting BERT inference for run %s on %s text(s)",
            request.run_id,
            len(request.texts),
        )
        result = await asyncio.to_thread(self._run_bert_inference_sync, request)
        activity.logger.info(
            "Completed BERT inference for run %s",
            request.run_id,
        )
        return result


# -------------------------------------------------------------------------------
# Checkpointing Activities
# -------------------------------------------------------------------------------
class BertCheckpointingActivities:
    @staticmethod
    def _create_dataset_snapshot_sync(request: DatasetSnapshotRequest) -> DatasetSnapshotResult:
        """Create a dataset snapshot synchronously.This helper is synchronous;
        the async activity delegates to it via ``asyncio.to_thread`` for non-blocking execution.
        """
        # Load the dataset
        raw_datasets = load_dataset(
            request.dataset_name,
            request.dataset_config,
            trust_remote_code=True,  # Turn off to disable loading custom dataset scripts
        )

        train_dataset = raw_datasets["train"]

        # Apply sampling if requested
        if request.max_samples and request.max_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(request.max_samples))

        # Compute content hash
        hasher = hashlib.sha256()
        for idx in sorted(range(len(train_dataset))):
            example = train_dataset[idx]
            content = f"{example.get('sentence', '')}|{example.get('label', '')}"
            hasher.update(content.encode("utf-8"))

        data_hash = hasher.hexdigest()[:16]

        # Create snapshot ID
        snapshot_id = f"{request.dataset_name}-{request.dataset_config}-{data_hash}"

        # Path for this snapshot
        snapshot_path = Path(request.snapshot_dir) / snapshot_id

        # Check if snapshot already exists
        if snapshot_path.exists():
            activity.logger.info(f"Reusing existing snapshot {snapshot_id} (identical data)")

            # Load existing metadata
            with open(snapshot_path / "metadata.json") as f:
                existing_metadata = json.load(f)

            return DatasetSnapshotResult(
                snapshot_id=snapshot_id,
                dataset_name=request.dataset_name,
                dataset_config=request.dataset_config,
                num_train_samples=existing_metadata["num_samples"],
                num_eval_samples=0,
                data_hash=data_hash,
                snapshot_timestamp=existing_metadata["created_at"],
                snapshot_path=str(snapshot_path),
            )

        # Create new snapshot
        activity.logger.info(f"Creating new dataset snapshot: {snapshot_id}")
        snapshot_path.mkdir(parents=True, exist_ok=True)

        # Save dataset in JSONL format
        train_dataset.to_json(snapshot_path / "data.jsonl")

        # Save metadata
        metadata = {
            "snapshot_id": snapshot_id,
            "dataset_name": request.dataset_name,
            "dataset_config": request.dataset_config,
            "num_samples": len(train_dataset),
            "data_hash": data_hash,
            "created_at": datetime.utcnow().isoformat(),
            "created_by_run": request.run_id,
            "format_version": "1.0",
        }

        with open(snapshot_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        activity.logger.info(f"Snapshot saved: {len(train_dataset)} samples, hash {data_hash}")

        return DatasetSnapshotResult(
            snapshot_id=snapshot_id,
            dataset_name=request.dataset_name,
            dataset_config=request.dataset_config,
            num_train_samples=len(train_dataset),
            num_eval_samples=0,
            data_hash=data_hash,
            snapshot_timestamp=datetime.utcnow().isoformat(),
            snapshot_path=str(snapshot_path),
        )

    @staticmethod
    @activity.defn
    async def create_dataset_snapshot(request: DatasetSnapshotRequest) -> DatasetSnapshotResult:
        """Temporal activity that creates a versioned dataset snapshot."""
        activity.logger.info(
            "Creating dataset snapshot for %s/%s",
            request.dataset_name,
            request.dataset_config,
        )

        # Offload to thread to avoid blocking worker
        snapshot = await asyncio.to_thread(
            BertCheckpointingActivities._create_dataset_snapshot_sync, request
        )

        activity.logger.info(
            "Dataset snapshot ready: %s (hash: %s)", snapshot.snapshot_id, snapshot.data_hash
        )

        return snapshot
