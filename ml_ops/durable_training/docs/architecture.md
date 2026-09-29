# BERT Checkpointed Training – Architecture

This document explains how the `bert_checkpointing` package extends the baseline `bert_finetune` pattern with **dataset snapshots** and **checkpoint‑aware resumption**.

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
    - Start a long‑running fine‑tuning activity (`fine_tune_bert`) that:
      - Uses the dataset snapshot.
      - May resume from the latest known checkpoint path if available.
    - Track mid‑run checkpoints via the `update_checkpoint` signal.
    - Expose the latest checkpoint via the `get_latest_checkpoint` query.
  - Output: `BertFineTuneResult` (possibly reconstructed from a dict for robustness).

- `BertInferenceWorkflow`
  - Input: `BertInferenceRequest`.
  - Thin wrapper around the `run_bert_inference` activity.
  - Output: `BertInferenceResult`.

### Activities

- `BertCheckpointingActivities`
  - `_create_dataset_snapshot_sync`:
    - Loads a dataset using `load_dataset(dataset_name, dataset_config)`.
    - Optionally subsamples it based on `max_samples`.
    - Computes a stable content hash from examples.
    - Writes a JSONL snapshot + metadata under `./data_snapshots/{snapshot_id}`.
    - Reuses an existing snapshot directory if the hash matches.
  - `create_dataset_snapshot` (async):
    - Offloads the sync helper via `asyncio.to_thread`.
    - Logs start/end and returns `DatasetSnapshotResult`.

- `BertFineTuneActivities`
  - Methods:
    - `compute_metrics` – returns accuracy from logits and labels.
    - `tokenize_function` – tokenizes the `"sentence"` field using the configured `max_seq_length`.
    - `_fine_tune_bert_sync` – the core training routine:
      - Picks device (CUDA / MPS / CPU).
      - Loads data:
        - From a JSONL snapshot if `request.dataset_snapshot` is set.
        - Directly from the Hugging Face dataset otherwise.
      - Tokenizes with `tokenize_function`.
      - Applies `max_train_samples` / `max_eval_samples`.
      - Builds `AutoModelForSequenceClassification`.
      - Configures `TrainingArguments` with:
        - `output_dir="./bert_runs/{run_id}"`.
        - `save_strategy="steps"` and `save_steps` tuned from dataset size.
      - Determines `resume_path`:
        - Prefer `request.resume_from_checkpoint` (from workflow state).
        - Else detect latest `checkpoint-*` under `output_dir`.
      - Runs `Trainer.train(resume_from_checkpoint=resume_path)`.
      - Optionally evaluates (`Trainer.evaluate()`).
      - Saves model + tokenizer to `output_dir`.
      - Counts mid‑run checkpoints and returns a rich `BertFineTuneResult`.
  - `fine_tune_bert` (async):
    - Offloads `_fine_tune_bert_sync` via `asyncio.to_thread`.
    - Sends heartbeats.
    - Streams `CheckpointInfo` updates to the training workflow via the `update_checkpoint` signal.

- `BertInferenceActivities`
  - `_run_bert_inference_sync`:
    - Loads tokenizer + model from `./bert_runs/{run_id}`.
    - Runs batched inference and returns `BertInferenceResult`.
  - `run_bert_inference` (async):
    - Offloads the sync helper to a thread and logs start/end.

---

## State model

### Workflow state

- `CheckpointedBertTrainingWorkflow`
  - `latest_checkpoint: CheckpointInfo | None`
    - Updated by signals from `BertFineTuneActivities`.
  - `run_id: str | None`
    - Canonical run identifier used by both activities and external clients.

- `BertInferenceWorkflow`
  - Stateless aside from parameters; returns a `BertInferenceResult`.

### Signals and queries

- Signal:
  - `update_checkpoint(info: CheckpointInfo) -> None`
    - Receives an `epoch`, `step`, `loss`, `path`, and `timestamp`.
    - Stores the latest checkpoint pointer in `latest_checkpoint`.

- Query:
  - `get_latest_checkpoint() -> CheckpointInfo | None`
    - Exposes the latest checkpoint information for dashboards, CLIs, or external tools.

This pattern lets you:

- Tail workflows to checkpoint directory paths without touching the filesystem from workflow code.
- Implement “resume from last checkpoint” semantics across retries and restarts.

---

## Determinism

- All dataset and model I/O is confined to activities:
  - Workflows only exchange Pydantic models and primitive data.

- `workflow.unsafe.imports_passed_through()`:
  - `workflow.py` wraps imports from `custom_types` in this context manager, keeping type usage while preserving replay safety.

- Randomness and time:
  - Any non‑deterministic behavior (e.g., seeds, timestamps) is handled inside activities or encoded into models (`run_id`, snapshot metadata), not in workflows.

As a result:

- You can safely replay training workflows and inspect how they evolved.
- Determinism issues are contained to activities, which can be tested in isolation.

---

## Timeouts, retries, and idempotency

### Activity options

- `create_dataset_snapshot`
  - Called with `start_to_close_timeout=timedelta(minutes=10)` from the training workflow (and from higher‑level coordinators).
  - Idempotent by design:
    - If a snapshot with the same content hash exists, it is reused.

- `fine_tune_bert`
  - Called with `start_to_close_timeout=timedelta(hours=2)`.
  - Heartbeats:
    - Allow detection of stuck/failed runs.
    - Support heartbeat timeouts and cancellation.

- `run_bert_inference`
  - Called with `start_to_close_timeout=timedelta(minutes=10)`.

### Idempotency strategy

- **Business key = `run_id`**
  - Dataset snapshots include `run_id` in metadata but are keyed by content hash.
  - Training output is written to `./bert_runs/{run_id}`.

- **Dataset snapshots**
  - `DatasetSnapshotRequest` + content hashing ensure that identical data slices map to the same `snapshot_id`.
  - Snapshots can be reused across experiments if desired.

- **Checkpoint resume**
  - `resume_from_checkpoint` points to a specific checkpoint directory.
  - Retries that reuse this path avoid redoing prior epochs.

---

## Backpressure and scaling

- **Task queues**
  - `bert-checkpointing-task-queue`:
    - Hosts `CheckpointedBertTrainingWorkflow` and `BertInferenceWorkflow` plus their activities.
  - Higher‑level examples (like `bert_eval`) split training and evaluation across two queues; here everything lives on one queue to keep the demo simple.

- **Worker concurrency**
  - `worker.py` uses a `ThreadPoolExecutor` to offload ML work.
  - For real deployments, you may:
    - Increase executor size.
    - Run multiple workers on the same queue.
    - Separate workers with different resource profiles (CPU vs GPU).

---

## Failure modes and behavior

### Dataset snapshotting failure

- If `create_dataset_snapshot` fails:
  - Temporal can retry the activity; if content is consistent, a subsequent attempt will reuse the same snapshot directory.

### Training worker crash / restart

- The in‑flight `fine_tune_bert` activity is retried by Temporal according to the retry policy.
- Because:
  - The dataset snapshot path is stable.
  - The latest checkpoint path is captured in workflow state.
- A retried training run can resume from the last saved checkpoint instead of starting over.

### Inference worker crash / restart

- `run_bert_inference` is retried, re‑loading the same checkpointed model from disk.
- Evaluation is repeatable and side‑effect‑free on the dataset/model pair.

---

## Production path (high level)

`bert_checkpointing` is closer to a production pattern than `bert_finetune`:

- You can:
  - Use dataset snapshots to achieve reproducible experiments over evolving datasets.
  - Use checkpoint‑aware resume to save GPU time and survive cluster failures.
- To move from demo to production:
  - Deploy Temporal Server (Temporal Cloud or self‑hosted).
  - Package workers into containers with:
    - Training workers on GPU nodes.
    - Inference/lightweight orchestration workers on CPU nodes.
  - Store metadata (`BertFineTuneResult`, `DatasetSnapshotResult`) in a persistent DB or experiment tracking system.

Higher‑level packages (`bert_eval`, `bert_sweeps`, etc.) then build on this foundation for multi‑config experiments and sweeps.
