# Competitive Comparison – BERT Checkpointed Training

This document compares the `bert_checkpointing` pattern—dataset snapshots + checkpoint‑aware BERT training on Temporal—to common orchestration alternatives.

---

## Temporal (this repo’s approach)

- **Durability semantics**
  - Workflows are durably recorded; each decision is in history.
  - Activities (snapshotting, training, inference) are retried according to policy.
  - Checkpoint information flows through signals and queries to allow **resume from last checkpoint**, not just restart from scratch.

- **Long‑running processes**
  - Fine‑tuning can run for hours; Temporal handles long‑lived workflows naturally.
  - Heartbeats and checkpoint signals provide:
    - Liveness tracking.
    - Mid‑run progress visibility.
    - A safe way to resume after failures.

- **Code‑first expressiveness**
  - Orchestration is pure Python (`workflow.py`), not YAML state machines.
  - Checkpoint handling, snapshot reuse, and resume behavior are expressed as normal control flow.

- **Deterministic replay**
  - Workflows remain deterministic; all nondeterministic work (I/O, randomness, ML libraries) lives in activities.
  - You can replay `CheckpointedBertTrainingWorkflow` to understand how and when checkpoints were created and used.

- **Portability**
  - Temporal is cloud‑agnostic; workers can run:
    - On‑prem GPU clusters.
    - Cloud GPU/CPU pools.
    - Hybrid setups.

- **Operational ergonomics**
  - Temporal Web gives a single pane of glass:
    - Training workflows.
    - Snapshotting activities.
    - Evaluation/inference workflows.
  - Signals and queries provide ergonomics for monitoring progress and checkpoints.

- **Scaling model**
  - Add workers to `bert-checkpointing-task-queue` to scale out horizontally.
  - Split queues (training vs eval) or models vs datasets as architecture grows.

---

## AWS Step Functions

- **Durability semantics**
  - Durable state machines with retries, but:
    - Dataset snapshots and checkpoints are your responsibility to design and track.
    - There is no built‑in notion of deterministic replay at the code level.

- **Checkpointed training**
  - You must:
    - Implement dataset snapshotting via S3 + Lambda/ECS.
    - Model checkpoint awareness in your own code.
    - Plumb checkpoint paths through Step Function state.

- **Complexity**
  - Complex, long‑running BERT experiments quickly produce large, hard‑to‑maintain JSON state machines.

---

## Azure Durable Functions

- **Durability semantics**
  - Good function‑level durability, similar in spirit to Temporal.
  - However, the programming model is coupled to Azure Functions and its storage model.

- **Checkpointed training**
  - Checkpoint logic still lives entirely in your code.
  - Orchestration of “snapshot → train with checkpoints → eval” is possible but less explicitly separated into workflows/activities than in Temporal.

- **Portability**
  - Lock‑in to Azure’s function runtime and tooling.

---

## Apache Airflow

- **Durability semantics**
  - DAG and task states are persisted; retries are configurable.
  - Long‑running tasks (e.g., multi‑hour training) are possible but not the primary design goal.

- **Checkpointed training**
  - Checkpoint awareness must be implemented inside tasks/operators.
  - Tracking and reusing checkpoints across DAG runs is manual and error‑prone.

- **Use case fit**
  - Excellent for ETL and scheduled data pipelines.
  - Less ergonomic for interactive, checkpoint‑aware model training with signals and queries.

---

## Dagster / Prefect

- **Durability semantics**
  - Provide solid orchestrators with state, retries, and asset management (Dagster).

- **Checkpointed training**
  - Checkpoint and snapshot patterns can be built, but:
    - Not as directly tied to a workflow replay model as Temporal.
    - Signals/queries and mid‑run interaction semantics are more limited or higher‑level.

- **ML ergonomics**
  - Great for batch pipelines and analytics.
  - For fine‑grained, long‑running training with mid‑run checkpoints and human‑in‑the‑loop behaviors, Temporal’s workflow model is often a better fit.

---

## When to prefer this Temporal pattern

Use `bert_checkpointing` + Temporal when:

- You need **checkpoint‑aware retries and restarts** for long‑running BERT training.
- You want **dataset snapshots** that make experiments reproducible even as upstream datasets change.
- You care about a **clear separation** between deterministic orchestration (workflows) and nondeterministic ML work (activities).
- You want the ability to:
  - Inspect and query **latest checkpoints** while runs are still in flight.
  - Safely resume from mid‑run state after crashes or deploys.

More traditional orchestrators (Airflow, Dagster, Prefect) or cloud‑native state machines (Step Functions, Durable Functions) can be sufficient for simple or short‑lived jobs, but for **durable, checkpointed, experiment‑heavy training workloads**, the Temporal pattern in this repo provides:

- Stronger durability semantics.
- Explicit replayability.
- Cleaner state modeling.
- Better ergonomics for long‑running, evolving ML systems.
