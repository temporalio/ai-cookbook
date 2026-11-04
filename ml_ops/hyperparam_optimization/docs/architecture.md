# BERT Hyperparameter Sweeps – Architecture

This document explains how the `bert_sweeps` package extends the `bert_eval` and `bert_checkpointing` examples with **random** and **ladder/TPE‑style** hyperparameter sweeps.

At a high level:

- Individual training runs are still **checkpoint‑aware** and use **dataset snapshots** for reproducibility.
- Evaluation reuses the same deterministic evaluation pattern from `bert_eval`.
- New **sweep workflows** (`SweepWorkflow`, `LadderSweepWorkflow`) sit on top and coordinate many training + eval pipelines as trials in a single experiment.

---

## Components

### Workflows

- `CheckpointedBertTrainingWorkflow`
  - Input: `BertFineTuneConfig`.
  - Responsibilities:
    - Normalize a canonical `run_id`:
      - Use `config.run_id` if provided.
      - Else derive one from `workflow.info().run_id` and write it back into `config`.
    - Create or reuse a dataset snapshot by calling `create_dataset_snapshot`.
    - Run a long‑lived fine‑tuning activity (`fine_tune_bert`) that:
      - Uses the dataset snapshot when present.
      - May resume from the latest known checkpoint path if available.
    - Track mid‑run checkpoints via the `update_checkpoint` signal.
    - Expose the latest checkpoint via the `get_latest_checkpoint` query.
  - Output: `BertFineTuneResult` (reconstructed from a dict if necessary for robustness).

- `BertEvalWorkflow`
  - Input: `BertEvalRequest`.
  - Responsibilities:
    - Log the evaluation context (run ID, dataset, split).
    - Call `evaluate_bert_model` as an activity with appropriate timeouts.
    - Normalize the result into `BertEvalResult` (even if the data converter yields a plain dict).
  - Output: `BertEvalResult`.

- `CoordinatorWorkflow`
  - Input: `CoordinatorWorkflowInput` with one or more `CoordinatorWorkflowConfig` entries.
  - Responsibilities:
    - For each config:
      - Choose and propagate a canonical `run_id` across the top‑level config, fine‑tune config, and eval config.
      - Populate `evaluation_config.model_path` with `./bert_runs/{run_id}` if unset.
    - Start a child `CheckpointedBertTrainingWorkflow` per config on `bert-training-task-queue`.
    - After training completes, start a child `BertEvalWorkflow` per config on `bert-eval-task-queue`.
    - Return a list of `BertEvalResult` objects (one per config).

- `SweepWorkflow`
  - Input: `SweepRequest` (experiment ID, base `CoordinatorWorkflowConfig`, `SweepSpace`, trial count, concurrency, seed).
  - Responsibilities:
    - Use `workflow.random()` (seeded by `SweepRequest.seed`) to:
      - Sample hyperparameters from `SweepSpace` for each trial.
      - Generate deterministic per‑trial `run_id`s.
    - For each trial:
      - Clone and mutate the base `CoordinatorWorkflowConfig`.
      - Start a child `CoordinatorWorkflow` for that config.
    - Collect `BertEvalResult` outputs into `TrialResult` entries.
    - Return a leaderboard of trials sorted by score (e.g., accuracy).

- `LadderSweepWorkflow`
  - Input: `SweepRequest`.
  - Responsibilities:
    - Encode the ladder as a sequence of stages:
      - `(epochs, max_train_samples, keep_top_k, n_new_candidates)`.
    - For each stage:
      - Build or extend a population of configs:
        - Promote `keep_top_k` best trials from previous stages.
        - Add `n_new_candidates` proposed using a simple TPE‑inspired sampler that:
          - Treats high‑scoring trials as “good”.
          - Samples new candidates from distributions biased toward good regions of the search space.
      - Call `CoordinatorWorkflow` for each candidate (respecting `max_concurrency` via a semaphore).
      - Record `TrialResult` entries (config + eval metrics + normalized score).
    - Return the final stage’s leaderboard.

---

## Activities

All ML and I/O is confined to activities in `bert_activities.py`:

- `BertFineTuneActivities`
  - `_infer_text_fields` / `_infer_label_field_and_task`
    - Infer text / pair / label fields from dataset schema when not provided.
    - Detect whether the task is classification vs regression.
  - `tokenize_function`
    - Performs single‑text or text‑pair tokenization depending on inferred fields.
  - `_fine_tune_bert_sync`
    - Picks device (CUDA / MPS / CPU).
    - Loads the dataset:
      - From a JSONL snapshot if `dataset_snapshot` is provided.
      - Directly from Hugging Face otherwise.
    - Applies sub‑sampling (`max_train_samples`, `max_eval_samples`) with optional pre‑shuffle.
    - Configures `AutoModelForSequenceClassification` and `Trainer`.
    - Saves checkpoints periodically and pushes `CheckpointInfo` into a queue.
    - Computes metrics and returns a rich `BertFineTuneResult`.
  - `fine_tune_bert` (activity)
    - Offloads the sync helper via `asyncio.to_thread`.
    - Sends heartbeats and streams checkpoint signals back to `CheckpointedBertTrainingWorkflow`.

- `BertCheckpointingActivities`
  - `_create_dataset_snapshot_sync`
    - Loads the raw dataset (or split).
    - Optionally sub‑samples based on `max_samples`.
    - Computes a stable content hash and writes a JSONL snapshot under `./data_snapshots/{snapshot_id}`.
    - Reuses existing snapshots when the content hash matches.
  - `create_dataset_snapshot` (activity)
    - Offloads to a background thread and returns `DatasetSnapshotResult`.

- `BertEvalActivities`
  - `_evaluate_bert_model_sync`
    - Loads a saved model + tokenizer from `model_path` (typically `./bert_runs/{run_id}`).
    - Runs batched inference on a Hugging Face dataset split.
    - Computes accuracy and other basic metrics.
  - `evaluate_bert_model` (activity)
    - Offloads to a background thread and returns `BertEvalResult`.

---

## State model

### Workflow state

- `CheckpointedBertTrainingWorkflow`
  - `latest_checkpoint: CheckpointInfo | None`
    - Updated via signals whenever a new checkpoint is saved.
  - `run_id: str | None`
    - Canonical run identifier shared with activities and external clients.

- `CoordinatorWorkflow`
  - `run_ids: list[str]`
    - Tracks canonical run IDs for each config.
  - `run_pointers: list[Awaitable[BertFineTuneResult]]`
    - Pointers to child training workflows.
  - `eval_pointers: list[Awaitable[BertEvalResult]]`
    - Pointers to child eval workflows.

- `SweepWorkflow` / `LadderSweepWorkflow`
  - Use local variables plus `TrialResult` structures to track:
    - The sampled configs (`fine_tune_config` + eval config).
    - The evaluation score for each trial.
  - `LadderSweepWorkflow` additionally maintains:
    - A `history: list[_TrialObs]` summarizing past trials for TPE sampling.

### Signals and queries

- `CheckpointedBertTrainingWorkflow`
  - Signal:
    - `update_checkpoint(info: CheckpointInfo) -> None`
  - Query:
    - `get_latest_checkpoint() -> CheckpointInfo | None`

Sweep workflows rely on these primitives indirectly through child workflows and activities.

---

## Determinism

- All non‑deterministic operations (dataset I/O, model I/O, random seeds inside training) are confined to activities.
- Workflows import Pydantic types inside `workflow.unsafe.imports_passed_through()` to keep replay behavior safe.
- Randomness inside sweep workflows uses `workflow.random()`:
  - Seeded via `SweepRequest.seed`.
  - Ensures that replays and re‑executions with the same input produce the same sequence of candidate configs.

As a result, you can:

- Re‑run a sweep with the same `SweepRequest` and obtain identical trial ordering and configs.
- Replay workflows in Temporal Web to debug orchestration logic without re‑running ML code.

---

## Timeouts, retries, and idempotency

- Activity options (see `workflows.py`):
  - `create_dataset_snapshot`
    - `start_to_close_timeout=timedelta(minutes=10)`.
    - Idempotent: reuses existing snapshots based on content hash.
  - `fine_tune_bert`
    - `start_to_close_timeout=timedelta(hours=2)` for realistic training runs.
    - Uses heartbeats so stuck activities can be detected and retried.
  - `evaluate_bert_model`
    - `start_to_close_timeout=timedelta(minutes=10)`.

- Idempotency strategy:
  - **Business key**: `run_id` (per training + eval pipeline).
  - Snapshots are keyed by dataset name/config + hash so they can be safely reused.
  - Model output is written to `./bert_runs/{run_id}`.
  - Repeated runs with the same `run_id` and snapshot are safe and predictable.

---

## Backpressure and scaling

- **Task queues**
  - `bert-training-task-queue`
    - Host `CheckpointedBertTrainingWorkflow` child workflows plus training and snapshot activities.
  - `bert-eval-task-queue`
    - Hosts `BertEvalWorkflow`, `CoordinatorWorkflow`, `SweepWorkflow`, and `LadderSweepWorkflow` plus eval activities.

- **Worker concurrency**
  - Workers use `ThreadPoolExecutor` for ML activities and explicitly limit `max_concurrent_activities` to keep local demos stable.
  - For real deployments, you can:
    - Increase or shard workers on each queue.
    - Separate workers onto dedicated GPU vs CPU pools.

- **Sweep concurrency**
  - `SweepRequest.max_concurrency` limits how many full training + eval pipelines run in parallel.
  - `LadderSweepWorkflow` uses an `asyncio.Semaphore` to respect this limit even when many candidates are queued.

---

## Failure modes and behavior

- **Dataset snapshot failure**
  - Retries will either:
    - Recreate the snapshot (if the first attempt failed mid‑write).
    - Reuse an existing snapshot when the content hash matches.

- **Training worker crash / restart**
  - A running `fine_tune_bert` activity is retried according to Temporal’s retry policy.
  - The workflow state still holds the latest checkpoint path, allowing resumed training.

- **Eval worker crash / restart**
  - `evaluate_bert_model` is re‑run using the same `model_path` and dataset split.
  - Evaluation is pure and side‑effect‑free with respect to model and dataset.

- **Sweep workflow crash / replay**
  - Because sweep workflows are deterministic and side‑effect‑free, Temporal can:
    - Replay them from history to reconstruct state.
    - Resume orchestration without re‑doing completed training or evaluation work.

---

## Production path

`bert_sweeps` is much closer to a production‑style experiment harness than a simple demo:

- You can:
  - Run multi‑stage sweeps over many models and datasets.
  - Persist snapshots and checkpoints for full reproducibility.
  - Integrate with experiment‑tracking systems by persisting `SweepResult` and `TrialResult` metadata.
- To make this production‑ready:
  - Deploy Temporal Server (either Temporal Cloud or self‑hosted).
  - Containerize:
    - Training workers with GPU access.
    - Eval/sweep workers on CPU or mixed pools.
  - Store results, metrics, and lineage (dataset hash, config hash, code version) in a durable database or experiment tracker.
