<!--
description: Build durable custom hyperparameter
optimzaiton that saves on GPU costs and improves performance.
tags:[ML Ops, hyperparameter optimization, autoML, python]
priority: 399
-->

# BERT Hyperparameter Sweeps (Temporal + Transformers)
This folder contains a demo that uses Temporal to orchestrate **checkpoint‑aware BERT fine‑tuning, evaluation, and hyperparameter sweeps** on top of Hugging Face Transformers and Datasets.

It builds directly on the patterns from the durable training module for checkpoint-aware durable training 
and adds **random and TPE‑style ladder sweeps** that explore a search space of hyperparameters in a durable, replay‑safe way.

---

## What this package demonstrates

- **Durable hyperparameter sweeps** where each trial is a full training + eval pipeline.
- **Checkpoint‑aware training** with dataset snapshots and mid‑run checkpoint signals.
- **Ladder + TPE‑style search** implemented as pure Temporal workflows.
- **Separation of concerns** between:
  - Training/snapshot activities on `bert-training-task-queue`.
  - Evaluation / sweep orchestration workflows on `bert-eval-task-queue`.

---

## Quickstart

From the project root (`hyperparam-optimization/`):

1. **Start Temporal Server** (if not already running):

   ```bash
   temporal server start-dev
   ```

2. **Start the training worker** (GPU / capable machine recommended):

   ```bash
   uv sync --dev
   uv run -m training_worker
   ```

3. **Start the evaluation / sweep worker** (CPU is fine):

   ```bash
   uv run -m worker
   ```

4. **Run the ladder sweep starter** in another terminal:

   ```bash
   uv run -m starter
   ```

   This:

   - Builds a `SweepRequest` (`ladder_config_1`) with a base `CoordinatorWorkflowConfig`
     and `SweepSpace` search ranges.
   - Starts `LadderSweepWorkflow` on `bert-eval-task-queue`.
   - Under the hood, each stage of the ladder:
     - Calls `CoordinatorWorkflow`, which:
       - Starts a `CheckpointedBertTrainingWorkflow` child for each config on `bert-training-task-queue`.
       - After training, starts a `BertEvalWorkflow` child per config.
     - Records `TrialResult` entries and uses them to propose new configs at later rungs.
   - Prints a concise summary of the best runs (dataset, split, number of examples, accuracy).

---

## Durability demo

To see Temporal’s durability across sweeps:

1. Start both workers and run the `starter` script as above.
2. After several trials have started (you see logs from `BertFineTuneActivities`), **kill the training worker** (Ctrl‑C).
3. Restart the training worker:

   ```bash
   uv run -m src.workflows.train_tune.bert_sweeps.training_worker
   ```

4. Observe in Temporal Web and logs that:
   - In‑flight `checkpointed` training children resume from the latest checkpoint (using `CheckpointInfo` signals).
   - The ladder sweep continues from the last completed trial, not from scratch.
   - The final leaderboard is still coherent.

Because all dataset/model I/O is inside activities and all orchestration logic is inside workflows, the system can be **replayed and evolved safely** while preserving correctness.

---

## Why Temporal (for this example)

Compared to traditional orchestration tools, Temporal is a strong fit for this kind of work:

- **Durability semantics**
  - Each trial is a workflow with exactly‑once state progression and automatic retries.
  - Checkpointed training can resume from mid‑run checkpoints instead of restarting.
- **Long‑running sweeps**
  - Ladder sweeps can run for hours or days across many trials without losing state.
  - Human‑in‑the‑loop or external signals could be added without changing the core design.
- **Code‑first expressiveness**
  - The ladder/TPE logic is plain Python (`LadderSweepWorkflow`), not YAML or static DAGs.
  - Complex branching and adaptive search strategies are easy to express and test.
- **Deterministic replay**
  - All randomness in the sweep uses Temporal’s deterministic RNG (`workflow.random()`).
  - Workflows can be replayed for debugging or auditing without re‑running ML workloads.
- **Portability & scaling**
  - Workers are plain Python processes; you can run them locally, on Kubernetes, or on managed Temporal.
  - Training and eval workers can be scaled independently, including to GPU pools.

For a broader, rubric‑style comparison with alternatives, see
`src/workflows/train_tune/bert_sweeps/docs/competitive-comparison.md`.

---

## Repo map (local to this folder)

- `custom_types.py` – Pydantic models for:
  - Dataset snapshots and checkpoints (`DatasetSnapshot*`, `CheckpointInfo`).
  - Training/eval configs and results (`BertFineTuneConfig`, `BertEvalRequest`, `BertEvalResult`).
  - Inference types (if you later add inference flows).
  - Coordinator and sweep types (`CoordinatorWorkflowConfig`, `CoordinatorWorkflowInput`, `SweepSpace`, `SweepRequest`, `SweepResult`, `TrialResult`).
- `bert_activities.py` – Activities for:
  - Snapshotting datasets into content‑addressed directories.
  - Schema‑aware, checkpoint‑aware fine‑tuning (`BertFineTuneActivities`).
  - Dataset evaluation of fine‑tuned checkpoints (`BertEvalActivities`).
  - Utility activities such as `set_seed` for reproducible experiments.
- `workflows.py` – Temporal workflows:
  - `CheckpointedBertTrainingWorkflow` – checkpoint‑aware training with dataset snapshots.
  - `BertEvalWorkflow` – evaluation of a fine‑tuned run on a dataset split.
  - `CoordinatorWorkflow` – orchestrates training + eval for one or more configs.
  - `SweepWorkflow` – simple random hyperparameter sweep.
  - `LadderSweepWorkflow` – staged, TPE‑style ladder sweep over `SweepSpace`.
- `worker.py` – Worker hosting orchestration workflows (`BertEvalWorkflow`, `CoordinatorWorkflow`, `CheckpointedBertTrainingWorkflow`, `SweepWorkflow`, `LadderSweepWorkflow`) plus evaluation activities on `bert-eval-task-queue`.
- `training_worker.py` – Worker hosting training and snapshot activities on `bert-training-task-queue`.
- `starter.py` – CLI entrypoint that kicks off `LadderSweepWorkflow` with a sample configuration and prints results.
- `tests/` – Workflow tests using Temporal’s `WorkflowEnvironment` to validate determinism and orchestration behavior.

For a deeper architectural breakdown, see:

- `src/workflows/train_tune/bert_sweeps/docs/architecture.md`

For a competitive comparison with other orchestrators, see:

- `src/workflows/train_tune/bert_sweeps/docs/competitive-comparison.md`
