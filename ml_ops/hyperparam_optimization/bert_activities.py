"""Temporal activities for BERT checkpointing and fine-tuning."""

import asyncio
import contextlib
import hashlib
import json
import os
import queue
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import torch

try:
    from datasets import ClassLabel
except Exception:
    ClassLabel = None
from datasets import load_dataset
from temporalio import activity
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from src.workflows.train_tune.bert_sweeps.custom_types import (
    BertEvalRequest,
    BertEvalResult,
    BertFineTuneRequest,
    BertFineTuneResult,
    CheckpointInfo,
    DatasetSnapshotRequest,
    DatasetSnapshotResult,
)

# Human-friendly error message surfaced when ML dependencies are missing. This keeps
# the Temporal worker process healthy even if the Python environment is not configured
# for running the BERT example.
TRANSFORMERS_IMPORT_MESSAGE: Final[str] = (
    "BERT checkpointing dependencies are not installed. "
    "Install 'transformers', 'datasets', and 'torch' to execute this activity."
)

# How frequently the fine-tuning activity should send heartbeats while training is running in a background thread.
HEARTBEAT_INTERVAL_SECONDS: Final[float] = 5.0


# -------------------------------------------------------------------------------
# Fine Tuning Activities
# -------------------------------------------------------------------------------
class BertFineTuneActivities:
    """Activity collection for checkpoint-aware BERT fine-tuning."""

    def __init__(self) -> None:
        self.config = None
        self.tokenizer = None
        self.text_field: str | None = None
        self.text_pair_field: str | None = None
        self.label_field: str | None = None
        self.task_type: str | None = None  # "classification" | "regression"
        self.num_labels: int | None = None

    def _infer_text_fields(self, sample: dict) -> None:
        """Infer (text_field, text_pair_field) from config overrides or dataset columns."""
        # 1) Config overrides win.
        if getattr(self.config, "text_field", None):
            self.text_field = self.config.text_field
            self.text_pair_field = getattr(self.config, "text_pair_field", None)
            return

        COMMON_TEXT_COLS = (
            "text",
            "sentence",
            "content",
            "review",
            "question",
            "article",
            "prompt",
        )
        COMMON_PAIR_COLS = (
            ("sentence1", "sentence2"),
            ("premise", "hypothesis"),
            ("question", "context"),
            ("query", "passage"),
        )

        # 2) Common pair schemas.
        for a, b in COMMON_PAIR_COLS:
            if (
                a in sample
                and b in sample
                and isinstance(sample[a], str)
                and isinstance(sample[b], str)
            ):
                self.text_field, self.text_pair_field = a, b
                return

        # 3) Common single text field names.
        for c in COMMON_TEXT_COLS:
            if c in sample and isinstance(sample[c], str):
                self.text_field = c
                self.text_pair_field = None
                return

        # 4) Fallback: first string field.
        for k, v in sample.items():
            if isinstance(v, str):
                self.text_field = k
                self.text_pair_field = None
                return

        raise KeyError(f"Couldn't infer a text column from dataset columns: {list(sample.keys())}")

    def _infer_label_field_and_task(self, train_features, sample: dict) -> None:
        """Infer label column, task type, and num_labels (or use config overrides)."""
        # 1) Config override for label field
        self.label_field = getattr(self.config, "label_field", None)

        # If not provided, try common names.
        if self.label_field is None:
            for c in ("label", "labels", "target", "score", "y"):
                if c in sample:
                    self.label_field = c
                    break

        if self.label_field is None:
            # Last-resort fallback: try any numeric scalar column
            for k, v in sample.items():
                if isinstance(v, (int, float)) and k not in (self.text_field, self.text_pair_field):
                    self.label_field = k
                    break

        if self.label_field is None:
            raise KeyError(
                f"Couldn't infer a label column from dataset columns: {list(sample.keys())}"
            )

        # 2) Infer task type (or honor config override)
        cfg_task = getattr(self.config, "task_type", "auto")
        feature = train_features.get(self.label_field)

        if cfg_task in ("classification", "regression"):
            self.task_type = cfg_task
        # Auto mode: if ClassLabel -> classification, if float -> regression, else classification.
        elif ClassLabel is not None and isinstance(feature, ClassLabel):
            self.task_type = "classification"
        else:
            v = sample[self.label_field]
            self.task_type = "regression" if isinstance(v, float) else "classification"

        # 3) Infer num_labels
        if self.task_type == "regression":
            self.num_labels = 1
            return

        # classification
        if ClassLabel is not None and isinstance(feature, ClassLabel):
            self.num_labels = int(feature.num_classes)
        else:
            # simple heuristic: gather a small set of unique labels from the first ~1k examples
            # (keeps it simple; avoids scanning the whole dataset)
            self.num_labels = None  # caller can fill using dataset slice if desired

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        logits = np.asarray(logits)
        labels = np.asarray(labels)

        # Normalize label shape to (N,)
        if labels.ndim > 1:
            labels = labels.reshape(-1)

        # Regression: common HF convention is num_labels == 1
        if (
            getattr(self.config, "num_labels", None) == 1
            or logits.ndim == 1
            or logits.shape[-1] == 1
        ):
            preds = logits.reshape(-1)
            mse = float(np.mean((preds - labels) ** 2))
            rmse = float(np.sqrt(mse))
            return {"mse": mse, "rmse": rmse}

        # Classification
        preds = np.argmax(logits, axis=-1)
        acc = float(np.mean(preds == labels))

        metrics = {"accuracy": acc}

        # Add a simple binary F1 when it looks binary
        if logits.shape[-1] == 2:
            tp = float(np.sum((preds == 1) & (labels == 1)))
            fp = float(np.sum((preds == 1) & (labels == 0)))
            fn = float(np.sum((preds == 0) & (labels == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            )

            metrics.update({"precision": precision, "recall": recall, "f1": f1})

        return metrics

    def tokenize_function(self, batch: dict) -> dict:
        if self.tokenizer is None or self.config is None:
            raise RuntimeError("Tokenizer/config not initialized; call _fine_tune_bert_sync first.")

        # Single-text or text-pair tokenization depending on what we inferred.
        # Be defensive in case we're mapping over a split whose schema differs
        # from the one we originally inspected.
        text_field = self.text_field
        text_pair_field = self.text_pair_field

        if text_field not in batch or (
            text_pair_field is not None and text_pair_field not in batch
        ):
            # Re-infer from the current batch's first example.
            sample: dict = {}
            for k, v in batch.items():
                # `v` is typically a list/array of values for this column.
                if isinstance(v, (list, tuple)) and v:
                    sample[k] = v[0]
                else:
                    sample[k] = v

            self._infer_text_fields(sample)
            text_field = self.text_field
            text_pair_field = self.text_pair_field

        if text_pair_field is not None:
            return self.tokenizer(
                batch[text_field],
                batch[text_pair_field],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length,
            )

        return self.tokenizer(
            batch[text_field],
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
        )

    def _cast_labels(self, batch: dict) -> dict:
        ys = batch["labels"]
        if self.task_type == "regression":
            return {"labels": [float(y) for y in ys]}
        return {"labels": [int(y) for y in ys]}

    def _fine_tune_bert_sync(
        self,
        request: BertFineTuneRequest,
        checkpoint_queue: "queue.Queue[CheckpointInfo] | None" = None,
    ) -> BertFineTuneResult:
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

        if self.config.seed is not None:
            set_seed(self.config.seed)
        else:
            set_seed(42)

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
        # ------------------------------------------------------------------
        schema_split_name = "train" if "train" in raw_datasets else next(iter(raw_datasets.keys()))
        schema_split = raw_datasets[schema_split_name]

        schema_sample = schema_split[0]  # just inspect columns/types

        # Infer the primary text field for this dataset (e.g. "sentence" for GLUE,
        # "text" for IMDB) so tokenization works across multiple sources.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        except Exception:
            activity.logger.warning(
                "Falling back to slow tokenizer for %s", self.config.model_name, exc_info=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=False)

        # Infer Text
        self._infer_text_fields(schema_sample)

        # Infer label field + task type (+ maybe num_labels)
        self._infer_label_field_and_task(schema_split.features, schema_sample)

        # If classification and we couldn't get num_labels from features, estimate cheaply
        if self.task_type == "classification" and self.num_labels is None:
            probe_n = min(1000, len(schema_split))
            label_probe = schema_split.select(range(probe_n))[self.label_field]
            self.num_labels = len(set(label_probe))

        # Apply the tokenizer across the dataset; `batched=True` lets HF process
        # multiple rows at once for better throughput.
        tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)

        # Normalize label column to "labels" for Trainer
        if self.label_field != "labels":
            tokenized_datasets = tokenized_datasets.rename_column(self.label_field, "labels")

        tokenized_datasets = tokenized_datasets.map(self._cast_labels, batched=True)
        # Tell Datasets to yield PyTorch tensors for the columns the Trainer needs.
        tokenized_datasets.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        # ------------------------------------------------------------------
        # 3. Prepare train and eval datasets, applying any sub-sampling requested.
        eval_dataset = (
            tokenized_datasets.get("validation")
            or tokenized_datasets.get("validation_matched")
            or tokenized_datasets.get("dev")
            or tokenized_datasets.get("val")
        )

        if eval_dataset is None and "train" in tokenized_datasets:
            split = tokenized_datasets["train"].train_test_split(
                test_size=0.1,
                seed=self.config.seed,
            )
            tokenized_datasets["train"] = split["train"]
            eval_dataset = split["test"]

        train_dataset = tokenized_datasets["train"]

        # ------------------------------------------------------------------
        # 3. Optionally sub-sample train/eval for a fast demo run on laptops.
        if self.config.shuffle_before_select:
            train_dataset = train_dataset.shuffle(seed=self.config.seed)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.shuffle(seed=self.config.seed)

        if self.config.max_train_samples is not None and self.config.max_train_samples < len(
            train_dataset
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
            num_labels=int(self.num_labels or 2),
            local_files_only=False,
            low_cpu_mem_usage=False,
            dtype=torch.float32,
        )

        p = next(model.parameters(), None)
        if p is not None and getattr(p, "is_meta", False):
            raise ApplicationError(
                "Model loaded as meta; accelerate/device_map path is still being taken.",
                non_retryable=True,
            )

        if self.task_type == "regression":
            model.config.problem_type = "regression"

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

        # Attach a callback that enqueues checkpoint metadata on each save, if requested.
        if checkpoint_queue is not None:
            trainer.add_callback(
                QueueingCheckpointCallback(
                    checkpoint_queue=checkpoint_queue,
                    num_epochs=self.config.num_epochs,
                ),
            )

        # Run training, but fall back to a fresh start if the checkpoint is
        # incompatible with the current model configuration (e.g., model_name
        # or num_labels changed between runs).
        try:
            train_result = trainer.train(resume_from_checkpoint=resume_path)
        except RuntimeError as exc:  # pragma: no cover - defensive path
            msg = str(exc)
            if "Error(s) in loading state_dict" in msg and "size mismatch for" in msg:
                activity.logger.warning(
                    "Incompatible checkpoint detected for run %s (likely model_name/label schema changed); "
                    "restarting training from scratch without resuming.",
                    request.run_id,
                )
                train_result = trainer.train()
            else:
                raise

        eval_metrics: dict[str, float] | None = None

        if eval_dataset is not None:
            raw_metrics = trainer.evaluate()

            # Keep only numeric scalars and normalize to plain floats
            eval_metrics = {
                k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))
            }

        # Persist the final fine-tuned model and tokenizer so that the inference
        # activity can load them later based solely on ``run_id``.
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        Path(training_args.output_dir, "READY").write_text("ok")

        # Discover any mid-run checkpoints created by the Trainer.
        checkpoint_dirs = sorted(p for p in output_dir.glob("checkpoint-*") if p.is_dir())
        total_checkpoints_saved = len(checkpoint_dirs)

        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        training_time_seconds = float(time.perf_counter() - start_time)

        return BertFineTuneResult(
            run_id=request.run_id,
            config=self.config,
            train_loss=float(train_result.training_loss),
            eval_metrics=eval_metrics,
            training_time_seconds=training_time_seconds,
            num_parameters=num_parameters,
            dataset_snapshot=request.dataset_snapshot,
            total_checkpoints_saved=total_checkpoints_saved,
            inferred_text_field=self.text_field,
            inferred_text_pair_field=self.text_pair_field,
            inferred_label_field=self.label_field,
            inferred_task_type=self.task_type,
            inferred_num_labels=self.num_labels,
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

        # Shared queue for checkpoint updates produced by the Trainer callback.
        checkpoint_queue: queue.Queue[CheckpointInfo] = queue.Queue()

        # Best-effort initialization of a workflow handle used for signaling.
        signal_handle = None
        try:
            info = activity.info()
            client = await Client.connect(
                "localhost:7233",
                data_converter=pydantic_data_converter,
            )
            signal_handle = client.get_workflow_handle(
                info.workflow_id,
                run_id=info.workflow_run_id,
            )
        except Exception:
            activity.logger.exception(
                "Failed to initialize checkpoint signaling; continuing without it",
            )

        training_task = asyncio.create_task(
            asyncio.to_thread(self._fine_tune_bert_sync, request, checkpoint_queue),
        )
        try:
            while not training_task.done():
                activity.heartbeat({"run_id": request.run_id})

                # Drain any newly produced checkpoints and signal them to the workflow.
                if signal_handle is not None:
                    while True:
                        try:
                            checkpoint_info = checkpoint_queue.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            await signal_handle.signal("update_checkpoint", checkpoint_info)
                        except Exception:
                            activity.logger.exception(
                                "Failed to signal checkpoint %s",
                                checkpoint_info.path,
                            )

                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            result = await training_task
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            training_task.cancel()
            with contextlib.suppress(Exception):
                await training_task
            raise

        # After training completes, flush any remaining checkpoints in the queue.
        if signal_handle is not None:
            try:
                while True:
                    checkpoint_info = checkpoint_queue.get_nowait()
                    try:
                        await signal_handle.signal("update_checkpoint", checkpoint_info)
                    except Exception:
                        activity.logger.exception(
                            "Failed to signal checkpoint %s",
                            checkpoint_info.path,
                        )
            except queue.Empty:
                pass

        # Log a concise summary of training metrics, handling both scalar and dict shapes.
        if not result.eval_metrics:
            eval_summary = "N/A"
        else:
            # prefer the common keys HF returns
            key_order = ("eval_accuracy", "accuracy", "eval_loss", "loss", "mse", "rmse", "f1")
            picked: list[str] = []
            for k in key_order:
                if k in result.eval_metrics:
                    picked.append(f"{k}={float(result.eval_metrics[k]):.4f}")
            if not picked:
                picked = [
                    f"{k}={float(v):.4f}"
                    for k, v in result.eval_metrics.items()
                    if isinstance(v, (int, float))
                ]
            eval_summary = ", ".join(picked) if picked else "N/A"

        activity.logger.info(
            "Completed BERT fine-tuning run %s with loss %.4f and metrics %s",
            request.run_id,
            result.train_loss,
            eval_summary,
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


# -------------------------------------------------------------------------------
# Evaluation Activities
# -------------------------------------------------------------------------------


class BertEvalActivities:
    """Activity collection for evaluating fine-tuned BERT models on public datasets."""

    @staticmethod
    def _evaluate_bert_model_sync(request: BertEvalRequest) -> BertEvalResult:
        """Evaluate a fine-tuned BERT model on a public dataset split.

        This helper loads a saved checkpoint from ``./bert_runs/{run_id}``, runs
        batched inference over a Hugging Face dataset (GLUE SST-2 by default),
        and computes simple accuracy. All I/O and ML details live here so the
        Temporal workflow layer can remain deterministic.
        """
        if torch is None or load_dataset is None:
            raise RuntimeError(TRANSFORMERS_IMPORT_MESSAGE)

        # Select device mirroring the training configuration.
        if request.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            request.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load the fine-tuned model and tokenizer from the configured model path.
        if not request.model_path:
            raise ValueError(
                "BertEvalRequest.model_path must be set by the coordinator workflow "
                "to locate the fine-tuned model checkpoint."
            )

        model_dir = Path(request.model_path).resolve()
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\nCurrent working directory: {Path.cwd()}\n"
            )

        ready = model_dir / "READY"
        if not ready.exists():
            raise ApplicationError(
                f"Model dir not finalized yet (missing READY): {model_dir}",
                non_retryable=False,  # allow Temporal retry
            )

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
        )

        model.to(device)

        if device.type in ("cuda", "mps"):
            model = model.to(dtype=torch.float16)

        model.eval()

        # Load the evaluation split from the requested dataset.
        raw_datasets = load_dataset(request.dataset_name, request.dataset_config_name)
        eval_dataset = raw_datasets[request.split]

        # Optionally subsample for fast, tutorial-friendly runs.
        if request.max_eval_samples is not None and request.max_eval_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(request.max_eval_samples))

        # Infer which text field(s) to use for tokenization in a way that
        # works across multiple datasets (GLUE, IMDB, etc.).
        sample = eval_dataset[0]
        text_field: str | None = None
        text_pair_field: str | None = None

        common_text_cols = (
            "text",
            "sentence",
            "content",
            "review",
            "question",
            "article",
            "prompt",
        )
        common_pair_cols = (
            ("sentence1", "sentence2"),
            ("premise", "hypothesis"),
            ("question", "context"),
            ("query", "passage"),
        )

        for a, b in common_pair_cols:
            if (
                a in sample
                and b in sample
                and isinstance(sample[a], str)
                and isinstance(sample[b], str)
            ):
                text_field, text_pair_field = a, b
                break

        if text_field is None:
            for c in common_text_cols:
                if c in sample and isinstance(sample[c], str):
                    text_field = c
                    break

        if text_field is None:
            for k, v in sample.items():
                if isinstance(v, str):
                    text_field = k
                    break

        if text_field is None:
            raise KeyError(
                f"Couldn't infer a text column from eval dataset columns: {list(sample.keys())}"
            )

        # Tokenize the evaluation dataset into tensors.
        def tokenize_batch(batch: dict) -> dict:
            if text_pair_field is not None:
                return tokenizer(
                    batch[text_field],
                    batch[text_pair_field],
                    padding="max_length",
                    truncation=True,
                    max_length=request.max_seq_length,
                )

            return tokenizer(
                batch[text_field],
                padding="max_length",
                truncation=True,
                max_length=request.max_seq_length,
            )

        tokenized_dataset = eval_dataset.map(tokenize_batch, batched=True)

        # Normalize label column to "labels" for the evaluation loop.
        label_column = "label" if "label" in eval_dataset.column_names else "labels"
        if label_column != "labels":
            tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")

        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        from torch.utils.data import DataLoader  # type: ignore[import]

        data_loader = DataLoader(
            tokenized_dataset,
            batch_size=request.batch_size,
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)

                outputs = model(**inputs)
                logits = outputs.logits
                preds = logits.argmax(dim=-1)

                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))

        accuracy = float(correct / total) if total > 0 else 0.0

        return BertEvalResult(
            run_id=request.run_id,
            dataset_name=request.dataset_name,
            dataset_config_name=request.dataset_config_name,
            split=request.split,
            num_examples=total,
            accuracy=accuracy,
        )

    @staticmethod
    @activity.defn
    async def evaluate_bert_model(request: BertEvalRequest) -> BertEvalResult:
        """Temporal activity that evaluates a fine-tuned BERT model on a dataset."""
        activity.logger.info(
            "Starting BERT evaluation for run %s on %s/%s[%s]",
            request.run_id,
            request.dataset_name,
            request.dataset_config_name,
            request.split,
        )

        result = await asyncio.to_thread(BertEvalActivities._evaluate_bert_model_sync, request)

        activity.logger.info(
            "Completed BERT evaluation for run %s: accuracy=%.3f over %s examples",
            result.run_id,
            result.accuracy,
            result.num_examples,
        )

        return result


# -------------------------------------------------------------------------------
# Checkpoint Callback Activities
# -------------------------------------------------------------------------------
class QueueingCheckpointCallback(TrainerCallback):
    """Trainer callback that enqueues checkpoint metadata for async signaling."""

    def __init__(self, checkpoint_queue: "queue.Queue[CheckpointInfo]", num_epochs: int) -> None:
        self._checkpoint_queue = checkpoint_queue
        self._num_epochs = num_epochs

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # Best-effort loss extraction: last logged loss in history (may be absent)
        loss = None
        for item in reversed(getattr(state, "log_history", []) or []):
            if "loss" in item:
                loss = float(item["loss"])
                break

        # Epoch can be float; make it int-ish for your CheckpointInfo
        epoch = int(state.epoch) if state.epoch is not None else self._num_epochs

        checkpoint_info = CheckpointInfo(
            epoch=epoch,
            step=int(state.global_step),
            path=ckpt_dir,
            loss=float(loss) if loss is not None else 0.0,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Enqueue the checkpoint info for the async activity wrapper to consume.
        try:
            self._checkpoint_queue.put_nowait(checkpoint_info)
        except Exception:
            # Never fail training because of queue issues.
            activity.logger.exception("Failed to enqueue checkpoint %s", ckpt_dir)

        return control


# -------------------------------------------------------------------------------
# Ladder Activities
# -------------------------------------------------------------------------------


@activity.defn(name="set_seed")
async def jitter_seed(seed: int) -> int:
    """Activity used by ladder sweeps to jitter a base seed deterministically.

    The activity name is kept as ``set_seed`` for compatibility with existing
    workflows, but the Python symbol is ``jitter_seed`` to avoid shadowing
    ``transformers.set_seed`` used in training activities.
    """
    seed = seed + random.randint(-10000, 10000)
    if seed <= 0:
        seed = random.randint(0, 20000)
    return seed
