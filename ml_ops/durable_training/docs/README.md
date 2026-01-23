# BERT Checkpointed Training (Temporal + Transformers)

This folder contains a demo that uses Temporal to orchestrate **checkpoint‑aware BERT fine‑tuning** and **resumable inference** on top of Hugging Face Transformers and Datasets.

At a high level:

- `workflow.py` defines `CheckpointedBertTrainingWorkflow` and `BertInferenceWorkflow` for deterministic orchestration.
- `bert_activities.py` owns all non‑deterministic ML work (dataset loading, tokenization, training, checkpointing, inference).
- `custom_types.py` holds the Pydantic models used by workflows, activities, and clients.
- `worker.py` hosts workflows and activities on the `bert-checkpointing-task-queue`.
- `starter.py` is a CLI-style entrypoint that runs a full training + inference demo.

## Quickstart

From the project root (`temporal_training/`):

1. **Start Temporal Server** (if not already running):

   ```bash
   temporal server start-dev
   ```

2. **Start the checkpointed BERT worker**:

   ```bash
   uv run -m src.workflows.train_tune.bert_checkpointing.worker
   ```

3. **Run the end-to-end checkpointed training + inference demo** in another terminal:

   ```bash
   uv run -m src.workflows.train_tune.bert_checkpointing.starter
   ```

   This:

   - Connects to Temporal using the Pydantic data converter.
   - Builds a `BertFineTuneConfig` with a fixed `run_id`.
   - Starts `CheckpointedBertTrainingWorkflow` on the `bert-checkpointing-task-queue`.
   - After training, runs `BertInferenceWorkflow` against the same run ID.
   - Prints a concise summary of training metrics and per‑text predictions.

## Durability demo

This example is designed to demonstrate Temporal’s durability around long‑running training:

1. Start the worker as above.
2. Run the `starter` script to kick off training.
3. After you see logs indicating that fine‑tuning has started, **kill the worker process** (Ctrl‑C).
4. Restart the worker:

   ```bash
   uv run -m src.workflows.train_tune.bert_checkpointing.worker
   ```

5. Observe in Temporal Web (or logs) that the workflow **resumes** from the last recorded checkpoint rather than starting from scratch.

The `CheckpointedBertTrainingWorkflow` tracks the latest checkpoint path via signals and passes it into the fine‑tuning activity so that retries and restarts can **resume** instead of re‑training from epoch 0.

## Why Temporal (for this example)

- **Durable training runs** – Long‑running fine‑tuning survives worker restarts without losing progress.
- **Deterministic orchestration** – All non‑deterministic ML logic stays in activities; workflows remain replay‑safe and easy to debug.
- **Checkpoint‑aware retries** – Retries can resume from mid‑run checkpoints instead of restarting from scratch, saving compute.
- **Introspection** – Signals and queries expose the latest checkpoint and run status for external tools or dashboards.

## Repo map (local to this folder)

- `custom_types.py` – Pydantic models for training configs, inference requests, checkpoint metadata, and results.
- `bert_activities.py` – Activities for dataset snapshotting, fine‑tuning, and inference.
- `workflow.py` – Checkpointed training + inference workflows.
- `worker.py` – Temporal worker that registers workflows and activities on `bert-checkpointing-task-queue`.
- `starter.py` – CLI entrypoint that runs a full training + inference demo.
- `tests/` – Unit and workflow tests for checkpointing behavior.

For broader project troubleshooting and Temporal background, see the root `README.md` and `docs/` directory.

For a step‑by‑step build guide showing how to assemble this module from scratch, see `src/workflows/train_tune/bert_checkpointing/CREATE_BERT_CHECKPOINTING.md`.

## Architecture

For a detailed breakdown of how checkpointed training is structured, see:

- `src/workflows/train_tune/bert_checkpointing/docs/architecture.md`

That document covers:

- How `CheckpointedBertTrainingWorkflow` and `BertInferenceWorkflow` interact with snapshotting and training activities.
- How `CheckpointInfo`, dataset snapshots, and `run_id` flow through the system.
- Determinism, timeouts, retries, and scaling considerations.

## Competitive comparison

To compare this checkpoint‑aware Temporal pattern against alternative orchestrators, see:

- `src/workflows/train_tune/bert_checkpointing/docs/competitive-comparison.md`

It evaluates this design against AWS Step Functions, Azure Durable Functions, Airflow, Dagster/Prefect, and others with respect to durability, long‑running workflows, replayability, portability, and operational ergonomics.
