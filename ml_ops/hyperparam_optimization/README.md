# BERT Hyperparameter Sweeps (Temporal + Transformers)

This folder contains a self-contained demo that uses Temporal to orchestrate
checkpointed BERT fine-tuning, evaluation, and hyperparameter sweeps on top of
Hugging Face Transformers and Datasets.

At a high level:

- **Activities** in `bert_activities.py` own all side effects: dataset loading,
  tokenization, model training, checkpointing, and evaluation.
- **Workflows** in `workflows.py` remain deterministic and focus purely on
  orchestration:
  - `CheckpointedBertTrainingWorkflow` runs a single fine-tuning job and emits
    checkpoints.
  - `BertEvalWorkflow` evaluates a fine-tuned checkpoint on a public dataset.
  - `CoordinatorWorkflow` wires training + evaluation together.
  - `SweepWorkflow` and `LadderSweepWorkflow` run hyperparameter sweeps.
- **Workers** in `worker.py` and `training_worker.py` split orchestration and
  heavy training onto separate task queues.
- **`starter.py`** is a tiny CLI entrypoint that kicks off a ladder-style sweep
  and prints a concise summary of results.

## Features

This workflow highlights Temporal capabilities, including:
- **Resource Isolation**: isolating code for infra and science teams into dedicated activity classes, separate from orchestration workflows or worker code.
- **Debug & Continue**: Debug crashes, bring down your workers, delpoy a new version, and continue right where your sweep left off without missing a beat
- **Dataset Checkpointing**: Datasets are persisted before training, allowing you to reproduce a past run exactly using only the information captured by temporal and the snapshotted dataset. This also ensures reproducibility is not lost if the dataset is later updated.
- **Flexible Sweeping System**: This system works arcross multiple models and datasets. Take a look at the sample configs in starter.py to see the different SLMs used during testing.

- **Live Updates**: Use signals to asynchronously update references to the latest checkpoints, dataset snapshots, etc during training. Here, we demonstrate how to do this by passing a ref to each new checkpoint to the Workflow, from the training activity. You can query this live or after workflow completion to locate the most up-to-date checkpoint.

- **TPE Based Scaling Ladders**: See how Temporal takes the stress away during large experimentation runs as a context manager, ensuring large and complicated orchestration happens reliably and consistently. Temporal is not limited to acyclic workflows like DAGs, and can be used to code looping behaviors, including loops with human in the loop steps, with ease.

This demo is just a starting point. Take a look and don't hesitate to ask questions.

## Prerequisites

- A running Temporal server (for local development, `temporal server start-dev`)
- Project dependencies installed:

  ```bash
  uv sync --dev
  ```

  The root `pyproject.toml` includes the ML dependencies (`transformers`,
  `datasets`, `torch`) required for this example.

- Optional but recommended: a GPU or Apple Silicon/MPS device for faster
  training. The example is configured to run on CPU as well, just more slowly.

## Files at a Glance

- `custom_types.py` – Pydantic models shared between workflows, activities,
  and clients (training configs, eval configs, sweep requests/results, etc.).
- `bert_activities.py` – Activities for:
  - Creating reproducible dataset snapshots.
  - Running checkpoint-aware BERT fine-tuning.
  - Evaluating fine-tuned checkpoints on public datasets.
- `workflows.py` – Temporal workflows for:
  - Single-run training/eval orchestration.
  - Random hyperparameter sweeps (`SweepWorkflow`).
  - Ladder-style sweeps with a simple TPE-inspired sampler
    (`LadderSweepWorkflow`).
- `worker.py` – Worker hosting evaluation and sweep workflows on the
  `bert-eval-task-queue`.
- `training_worker.py` – Worker hosting training activities on the
  `bert-training-task-queue`, suitable for GPU machines.
- `starter.py` – CLI script that builds a `SweepRequest`, runs
  `LadderSweepWorkflow`, and prints a table of results.

## Running the Demo (Ladder Sweep)

These steps assume you are in the project root (`temporal_training/`).

1. **Start Temporal Server** (if not already running):

   ```bash
   temporal server start-dev
   ```

2. **Start the training worker** (ideally on a machine with a GPU):

   ```bash
   uv run -m src.workflows.train_tune.bert_sweeps.training_worker
   ```

3. **Start the evaluation/sweep worker** (CPU-only is fine):

   ```bash
   uv run -m src.workflows.train_tune.bert_sweeps.worker
   ```

4. **Run the ladder sweep starter** in a third terminal:

   ```bash
   uv run -m src.workflows.train_tune.bert_sweeps.starter
   ```

   This:

   - Connects to the Temporal server using the Pydantic data converter.
   - Uses the sample `ladder_config_1` defined in `starter.py`, which targets
     the SciBERT + SciCite combination by default.
   - Starts a `LadderSweepWorkflow` execution on the `bert-eval-task-queue`.
   - Prints a small table summarizing each evaluated run (dataset, split,
     number of examples, accuracy).

5. **Inspect checkpoints and snapshots**:

   - Fine-tuned models and tokenizers are written under `./bert_runs/{run_id}`.
   - Dataset snapshots (if enabled) are written under `./data_snapshots`.

## Durability demo

To exercise Temporal’s durability with this sweep:

1. Start both workers (`training_worker` and `worker`) as described above.
2. Launch the ladder sweep via `starter.py`.
3. Once several trials are in flight, **kill the training worker** (Ctrl‑C).
4. Restart the training worker:
   ```bash
   uv run -m src.workflows.train_tune.bert_sweeps.training_worker
   ```
5. Observe in Temporal Web and logs that workflows continue from their last recorded state, and the sweep still completes with a coherent set of results.

Because all side effects live in activities, workflows can be safely replayed and retried without double‑applying updates.

## Customizing the Sweep

- Edit `ladder_config_1` in `starter.py` to:
  - Switch to a different base model or dataset.
  - Change the search space (`SweepSpace`) for learning rate, batch size,
    number of epochs, and sequence length.
  - Adjust `num_trials` and `max_concurrency` for your hardware.
- Tweak the `stages` list in `LadderSweepWorkflow.run` (see `workflows.py`) to
  change how many rungs the ladder uses and how much data/epochs each rung
  sees.
- For a simpler baseline, you can wire up `SweepWorkflow` instead of
  `LadderSweepWorkflow` to run a purely random sweep.

Because all randomness flows through Temporal's deterministic RNG, you can
re-run the same sweep with the same `SweepRequest.seed` and expect identical
trial configurations and ordering, which makes this a good template for
reproducible experimentation.

## Why Temporal (for this example)

- **Durable hyperparameter sweeps**: Long-running experiments survive crashes and worker restarts.
- **Deterministic experimentation**: Sweeps are repeatable because randomness flows through Temporal’s deterministic APIs.
- **Code-first orchestration**: Complex ladder/TPE behavior is expressed in Python workflows, not YAML.
- **Clear separation of concerns**: Activities own ML logic; workflows own orchestration and experiment structure.
