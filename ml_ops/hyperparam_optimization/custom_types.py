"""Shared data models for the BERT fine-tuning and inference example.

These Pydantic models are used in three places:

- As workflow inputs/outputs (e.g., ``BertExperimentInput``)
- As activity inputs/outputs (e.g., ``BertFineTuneRequest``)
- From external clients (see ``train.py`` and ``inference.py``) that submit
  workflows to Temporal.

Keeping the types in a dedicated module makes it easy to reuse them across
workflows, activities, and client code while preserving a single source of
truth for field names and validation rules.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Checkpointing - related types
# ---------------------------------------------------------------------------
class DatasetSnapshotRequest(BaseModel):
    """Request to create a dataset snapshot."""

    run_id: str = Field(description="Identifier for the training run requesting this snapshot.")

    dataset_name: str = Field(description="HuggingFace dataset name (e.g., 'glue').")

    dataset_config: str = Field(description="Dataset configuration name (e.g., 'sst2').")

    max_samples: int | None = Field(
        default=None, description="Optional limit on number of samples to include."
    )

    snapshot_dir: str = Field(
        default="./data_snapshots", description="Directory to store snapshots."
    )


class DatasetSnapshotResult(BaseModel):
    """Result describing a materialized dataset snapshot.

    This model mirrors the information persisted alongside each snapshot on disk
    so that workflows and activities can reason about snapshot identity and size
    without re-reading the underlying data.
    """

    snapshot_id: str = Field(
        description=(
            "Stable identifier for the snapshot, typically derived from the "
            "dataset name, configuration, and a content hash."
        ),
    )
    dataset_name: str = Field(
        description="Hugging Face dataset name (e.g., 'glue').",
    )
    dataset_config: str = Field(
        description="Dataset configuration name (e.g., 'sst2').",
    )
    num_train_samples: int = Field(
        description="Number of training samples stored in the snapshot.",
    )
    num_eval_samples: int = Field(
        description="Number of evaluation samples stored in the snapshot.",
    )
    data_hash: str = Field(
        description="Short content hash used to detect identical snapshots.",
    )
    snapshot_timestamp: str = Field(
        description="ISO 8601 timestamp indicating when the snapshot was created.",
    )
    snapshot_path: str = Field(
        description="Filesystem path where the snapshot artifacts are stored.",
    )
    dataset_schema: dict[str, str] | None = Field(
        default=None,
        description="Optional schema hints such as text_field/label_field/task_type/num_labels.",
    )


class CheckpointInfo(BaseModel):
    """Checkpoint information sent via signals."""

    epoch: int = Field(description="Training epoch number")
    step: int = Field(description="Global training step")
    path: str = Field(description="Filesystem path to checkpoint")
    loss: float = Field(description="Training loss at this checkpoint")
    timestamp: str = Field(description="ISO timestamp when saved")


# ---------------------------------------------------------------------------
# Activity-level types (training and inference)
# ---------------------------------------------------------------------------


class BertFineTuneConfig(BaseModel):
    """Configuration for a single BERT fine-tuning run.

    This mirrors the configuration accepted by the fine-tuning activity and is
    deliberately lightweight so it can be serialized in workflow inputs and
    reused by external clients.
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="Hugging Face model identifier to fine-tune.",
    )
    dataset_name: str = Field(
        default="glue",
        description="Hugging Face dataset name (e.g. 'glue').",
    )
    dataset_config_name: str = Field(
        default="sst2",
        description="Dataset configuration name (e.g. 'sst2').",
    )
    num_epochs: int = Field(
        gt=0,
        le=20,
        default=3,
        description="Number of training epochs.",
    )
    batch_size: int = Field(
        gt=0,
        le=128,
        default=16,
        description="Per-device training batch size.",
    )
    learning_rate: float = Field(
        gt=0,
        le=1e-1,
        default=5e-5,
        description="Learning rate for the optimizer.",
    )
    max_seq_length: int = Field(
        default=128,
        gt=8,
        le=512,
        description="Maximum sequence length for tokenization.",
    )
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU/MPS if available.",
    )

    max_train_samples: int | None = Field(
        default=2_000,
        description=(
            "Optional cap on the number of training examples to use. "
            "Set to None to train on the full dataset."
        ),
    )
    max_eval_samples: int | None = Field(
        default=1_000,
        description=(
            "Optional cap on the number of evaluation examples to use. "
            "Set to None to evaluate on the full validation set."
        ),
    )
    text_field: str | None = Field(default=None, description="Primary text column name.")
    text_pair_field: str | None = Field(
        default=None, description="Optional second text column for pair tasks."
    )
    label_field: str | None = Field(
        default=None, description="Label column name (e.g., 'label', 'labels', 'target')."
    )
    task_type: Literal["auto", "classification", "regression"] = Field(
        default="auto",
        description="Task type. 'auto' infers from dataset features.",
    )
    shuffle_before_select: bool = Field(
        default=True,
        description=(
            "Whether to shuffle the dataset before applying max_train_samples "
            "or max_eval_samples. Improves representativeness for small demo runs."
        ),
    )

    seed: int = Field(
        default=42,
        description="Random seed used for dataset shuffling and train/validation splits.",
    )

    run_id: str | None = Field(
        default=None, description="Logical identifier for this fine-tuning run."
    )


class BertFineTuneRequest(BaseModel):
    """Input to a BERT fine-tuning activity run."""

    run_id: str
    """Logical identifier for this fine-tuning run."""

    config: BertFineTuneConfig
    """Configuration for the fine-tuning experiment."""

    # NEW: Pass the dataset snapshot to use for reproducible training
    dataset_snapshot: DatasetSnapshotResult | None = Field(
        default=None, description="Optional dataset snapshot to use for reproducible training."
    )

    # NEW: Allow resuming from checkpoint through act
    resume_from_checkpoint: str | None = Field(
        default=None, description="Path to checkpoint to resume from (if retrying)"
    )


class BertFineTuneResult(BaseModel):
    """Summary metrics from a BERT fine-tuning run."""

    run_id: str
    """Echoed identifier for the run."""

    config: BertFineTuneConfig
    """Configuration actually used for the run."""

    train_loss: float = Field(
        description="Final training loss reported by the Trainer.",
    )

    eval_metrics: dict[str, float] | None = Field(
        default=None,
        description="Evaluation metrics reported by Trainer.evaluate() (e.g., accuracy, f1, mse).",
    )

    training_time_seconds: float = Field(
        description="Wall-clock training time in seconds.",
    )

    num_parameters: int = Field(
        description="Total number of learnable parameters in the model.",
    )

    dataset_snapshot: DatasetSnapshotResult | None = Field(
        default=None, description="Dataset snapshot used for this training run (if any)."
    )

    total_checkpoints_saved: int = Field(
        description="Total number of checkpoints saved during training.",
    )
    inferred_text_field: str | None = Field(default=None)
    inferred_text_pair_field: str | None = Field(default=None)
    inferred_label_field: str | None = Field(default=None)
    inferred_task_type: str | None = Field(default=None)  # "classification" / "regression"
    inferred_num_labels: int | None = Field(default=None)


class BertInferenceRequest(BaseModel):
    """Input to a BERT inference activity run."""

    run_id: str = Field(
        description="Identifier of the fine-tuned run whose checkpoint to load.",
    )
    texts: list[str] = Field(
        min_length=1,
        description="List of input sentences to classify.",
    )
    max_seq_length: int = Field(
        gt=8,
        le=512,
        default=128,
        description="Maximum sequence length for tokenization during inference.",
    )
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU/MPS for inference if available.",
    )


class BertInferenceResult(BaseModel):
    """Inference results for a batch of texts using a fine-tuned BERT model."""

    run_id: str
    """Identifier of the fine-tuned run used for inference."""

    texts: list[str]
    """Input texts that were classified."""

    predicted_labels: list[int] = Field(
        description="Predicted class indices for each input text.",
    )

    confidences: list[float] = Field(
        description="Confidence scores (max softmax probability) per prediction.",
    )


# ---------------------------------------------------------------------------
# Evaluation-level types
# ---------------------------------------------------------------------------


class BertEvalRequest(BaseModel):
    """Input configuration for evaluating a fine-tuned BERT model."""

    run_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the fine-tuned run whose checkpoint directory to load. "
            "May be left unset in high-level configs and filled in by a coordinator "
            "workflow once the training run ID is known."
        ),
    )
    dataset_name: str = Field(
        default="glue",
        description="Hugging Face dataset name to evaluate on (e.g., 'glue').",
    )
    dataset_config_name: str = Field(
        default="sst2",
        description="Dataset configuration name (e.g., 'sst2').",
    )
    split: str = Field(
        default="validation",
        description="Dataset split to evaluate on (e.g., 'validation' or 'test').",
    )
    max_eval_samples: int | None = Field(
        default=1_000,
        description=(
            "Optional cap on the number of evaluation examples to use. "
            "Set to None to evaluate on the full split."
        ),
    )
    max_seq_length: int = Field(
        gt=8,
        le=512,
        default=128,
        description="Maximum sequence length for tokenization during evaluation.",
    )
    batch_size: int = Field(
        gt=0,
        le=256,
        default=32,
        description="Batch size to use during evaluation.",
    )
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU/MPS for evaluation if available.",
    )

    model_path: str | None = Field(
        default=None,
        description=(
            "Path to the fine-tuned model checkpoint. Typically filled in by the "
            "coordinator workflow; if still unset at activity time, evaluation will fail."
        ),
    )

    seed: int = Field(default=42)


class BertEvalResult(BaseModel):
    """Aggregate metrics from evaluating a fine-tuned BERT model."""

    run_id: str = Field(
        description="Identifier of the fine-tuned run whose checkpoint was evaluated.",
    )
    dataset_name: str = Field(
        description="Hugging Face dataset name used for evaluation.",
    )
    dataset_config_name: str = Field(
        description="Dataset configuration name used for evaluation.",
    )
    split: str = Field(
        description="Dataset split that was evaluated (e.g., 'validation').",
    )
    num_examples: int = Field(
        ge=0,
        description="Number of examples included in the evaluation.",
    )
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Classification accuracy over the evaluated examples.",
    )

    baseline_accuracy: float | None = Field(
        default=None,
        description=(
            "Optional baseline accuracy used for ablation comparison for this result "
            "(for example, a lower-budget training configuration)."
        ),
    )
    improvement_vs_baseline: float | None = Field(
        default=None,
        description=(
            "Optional improvement in accuracy over the ablation baseline "
            "(accuracy - baseline_accuracy)."
        ),
    )


# ---------------------------------------------------------------------------
# Workflow-level types
# ---------------------------------------------------------------------------


class BertExperimentInput(BaseModel):
    """Input to the BERT fine-tuning workflow.

    The workflow fans out one fine-tuning activity per entry in ``runs`` and
    aggregates the resulting :class:`BertFineTuneResult` values.
    """

    experiment_name: str
    """Human-readable label for this experiment."""

    runs: list[BertFineTuneConfig]
    """List of fine-tuning configurations to execute."""


class BertExperimentOutput(BaseModel):
    """Summary of the BERT fine-tuning experiment."""

    experiment_name: str
    """Echoed experiment name."""

    runs: list[BertFineTuneResult]
    """Per-configuration fine-tuning results."""


class CoordinatorWorkflowConfig(BaseModel):
    """Input to the Coordinator Workflow."""

    run_id: str | None = Field(
        default=None, description="Recommended logical identifier for this fine-tuning run."
    )

    fine_tune_config: BertFineTuneConfig = Field(
        description="Configuration for the BERT fine-tuning run."
    )
    dataset_snapshot: DatasetSnapshotResult | None = Field(
        default=None, description="Optional dataset snapshot to use for reproducible training."
    )
    evaluation_config: BertEvalRequest = Field(
        description="Configuration for evaluating the fine-tuned BERT model."
    )


class CoordinatorWorkflowInput(BaseModel):
    """Input to the Coordinator Workflow."""

    configs: list[CoordinatorWorkflowConfig] = Field(
        description="List of configurations for the coordinator workflow to run and manage."
    )


# ---------------------------------------------------------------------------
# Sweep-level types
# ---------------------------------------------------------------------------


class SweepSpace(BaseModel):
    """Search space of hyperparameters used by sweep workflows.

    Defaults are intentionally small and laptop-friendly so that the example
    can run end-to-end in a few minutes. You can widen these ranges and lists
    when running larger experiments.
    """

    learning_rate: tuple[float, float] = (1e-5, 5e-5)  # log-uniform
    batch_size: list[int] = [2, 4, 8]
    max_seq_length: list[int] = [64, 128, 256]
    num_epochs: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


class SweepRequest(BaseModel):
    """Request to run a hyperparameter sweep over ``SweepSpace``."""

    experiment_id: str
    """Human-friendly identifier used to prefix per-trial run IDs."""

    base: CoordinatorWorkflowConfig
    """Base coordinator configuration that individual trials clone and mutate."""

    space: SweepSpace
    """Hyperparameter search space from which trials sample parameters."""

    num_trials: int = 8
    """Total number of trials to run across all rungs/stages."""

    max_concurrency: int = 4
    """Maximum number of training/eval pipelines to run in parallel."""

    seed: int = 42
    """Base random seed used to keep workflow-side sampling deterministic."""


class TrialResult(BaseModel):
    """Single trial result in a hyperparameter sweep."""

    run_id: str
    fine_tune_config: BertFineTuneConfig
    eval_result: BertEvalResult
    score: float  # normalized objective, higher is better


class SweepResult(BaseModel):
    """Aggregated results from a hyperparameter sweep."""

    experiment_id: str
    metric: str = "accuracy"
    best_run_id: str
    best_score: float
    leaderboard: list[TrialResult]
