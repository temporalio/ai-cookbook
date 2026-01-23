# Competitive Comparison – BERT Hyperparameter Sweeps

This document compares the `bert_sweeps` pattern (Temporal‑based hyperparameter sweeps with checkpointed training) against several common orchestration options.

The focus is on **long‑running, failure‑tolerant experiment sweeps** rather than simple batch jobs.

Compared systems:

- Temporal (this repo)
- AWS Step Functions
- Azure Durable Functions
- Airflow
- Dagster / Prefect
- (Briefly) Argo Workflows / Kubeflow Pipelines

---

## Rubric

We use the same rubric as the rest of the `train_tune` portfolio:

- **Durability semantics**
- **Long‑running processes**
- **Code‑first expressiveness**
- **Deterministic replay**
- **Portability**
- **Operational ergonomics**
- **Scaling model**
- **Cost & limits** (qualitative)

---

## Temporal (this repo)

- **Durability semantics**
  - Strong exactly‑once workflow state semantics.
  - Each training + eval pipeline is a workflow; sweeps are workflows composed of child workflows.
  - Automatic activity retries with backoff; timeouts and heartbeats configured in code.

- **Long‑running processes**
  - Training runs and sweeps can last hours or days.
  - History is stored durably; workers can crash and be replaced without losing progress.
  - Checkpoint‑aware training means retries can resume from mid‑run checkpoints.

- **Code‑first expressiveness**
  - Ladder + TPE logic is implemented entirely in Python (`LadderSweepWorkflow`).
  - Signals, queries, and child workflows enable rich patterns (e.g., adaptive sweeps, human‑in‑the‑loop).

- **Deterministic replay**
  - Workflows are deterministic:
    - All I/O and ML code lives in activities.
    - Randomness uses Temporal’s deterministic RNG.
  - You can replay a sweep to understand exactly how configs were chosen and which trials ran.

- **Portability**
  - Temporal Server can run in your own infra or via Temporal Cloud.
  - Workers are ordinary Python services and can run anywhere with network access.

- **Operational ergonomics**
  - Temporal Web shows workflow and activity histories, retries, and failures.
  - You can:
    - Cancel or reset individual trials.
    - Replay and inspect sweeps.
    - Start ad‑hoc experiments from the CLI or API.

- **Scaling model**
  - Horizontal scaling is via workers on named task queues.
  - You can independently scale:
    - Training workers on GPU queues.
    - Eval and sweep orchestration on CPU queues.

- **Cost & limits**
  - Costs track the amount of workflow history and number of tasks.
  - Efficient checkpointing (few, informative checkpoints) and careful activity timeouts help keep histories moderate.

---

## AWS Step Functions

- **Durability semantics**
  - Native support for long‑running state machines with strong durability.
  - However, per‑state limits and payload size constraints can make rich experiment metadata awkward.

- **Long‑running processes**
  - Long‑running workflows are possible but typically modeled as coarse states.
  - Fine‑grained control over per‑trial retries and checkpoint semantics is more cumbersome.

- **Code‑first expressiveness**
  - Primary model is declarative JSON/YAML state machines.
  - Implementing ladder/TPE logic usually requires embedding application code in Lambda or containers and wiring via states—less ergonomic and harder to test in isolation.

- **Deterministic replay**
  - State machine executions are recorded, but there is no built‑in concept equivalent to Temporal replay where the same code is re‑run from history for debugging.

- **Portability**
  - Fully tied to AWS.
  - Migrating an experiment harness to another cloud requires redesign.

- **Operational ergonomics**
  - Good console for visualizing executions, but:
    - Limited ability to refactor workflows while preserving history.
    - Harder to reuse the same code for local tests and cloud executions.

- **Scaling model**
  - Scaling is largely managed by AWS through Lambda / service integrations.
  - GPU‑heavy or custom training workloads usually require separate orchestration of EC2 / ECS / EKS.

- **Cost & limits**
  - Per‑step and duration pricing; complex sweeps with many small states can become expensive.
  - Payload size and history limits can affect designs that track many trials.

---

## Azure Durable Functions

- **Durability semantics**
  - Durable Functions provide an orchestration model with checkpoints and replay.
  - Similar in spirit to Temporal’s workflow replays but more tightly coupled to Azure Functions.

- **Long‑running processes**
  - Long‑running orchestrations are supported, but these are typically:
    - Written as orchestrator functions using special patterns.
    - Bound tightly to the Azure Functions runtime and storage.

- **Code‑first expressiveness**
  - Orchestrator code must obey Durable Functions constraints (e.g., no non‑deterministic I/O).
  - Implementing ladder/TPE behavior is possible but more constrained and less portable.

- **Deterministic replay**
  - Replay semantics are built in, but only within the Durable Functions environment.

- **Portability**
  - Strongly tied to Azure; not ideal if you want a cloud‑agnostic experiment harness.

- **Operational ergonomics**
  - Azure tooling is good for inspecting orchestrations, but:
    - Local testing and CI integration are more complex than running Temporal workers and using `WorkflowEnvironment`.

- **Scaling model**
  - Scaling is managed by Azure Functions; good for HTTP‑ and event‑driven workloads, but less tailored to GPU training clusters.

- **Cost & limits**
  - Consumption‑based pricing for functions; long‑running orchestrations and many activations can accumulate cost.

---

## Airflow

- **Durability semantics**
  - DAG runs are persisted in a metadata database.
  - Tasks can be retried, but there is no notion of a single, durable workflow state with replayable history.

- **Long‑running processes**
  - Airflow favors relatively short‑lived batch tasks.
  - Long‑running training jobs are often offloaded to external systems (Kubernetes, SageMaker, etc.) with callbacks.

- **Code‑first expressiveness**
  - DAGs are Python, but constrained by Airflow’s scheduling model.
  - Implementing interactive ladder/TPE logic requires custom operators and is not naturally stateful across many tasks.

- **Deterministic replay**
  - Re‑running a DAG creates a new run; Airflow doesn’t replay DAG code from history like Temporal.

- **Portability**
  - Airflow is self‑hosted and portable across clouds, but integration with cloud‑native training services often adds complexity.

- **Operational ergonomics**
  - Mature UI for DAG runs, but:
    - No built‑in idea of child workflows and per‑trial lineage.
    - Backfills and retries operate at DAG/task granularity, not at a “trial in a sweep” level.

- **Scaling model**
  - Scheduler + workers model; good for many small tasks, less ergonomic for long‑running GPU training with fine‑grained checkpoints.

- **Cost & limits**
  - Infra cost scales with scheduler/worker cluster.
  - No per‑task billing, but additional systems are often needed for durable checkpoints and experiment tracking.

---

## Dagster / Prefect

- **Durability semantics**
  - Both provide solid orchestration of tasks with retries and state tracking.
  - Task‑level durability is good, but long‑running, fine‑grained checkpointing usually requires manual integration with storage systems.

- **Long‑running processes**
  - Capable of orchestrating long jobs, but:
    - Patterns for multi‑stage, adaptive sweeps are less standardized.
    - Temporal’s workflow history and replay make it easier to model sweeps as first‑class workflows.

- **Code‑first expressiveness**
  - Strong Python APIs; easier than Airflow for expressing dynamic graphs.
  - Ladder/TPE can be implemented, but you still need to build your own abstractions for child workflows, signals, and queries.

- **Deterministic replay**
  - Runs are logged, but there is no built‑in, code‑level replay concept equivalent to Temporal’s workflows.

- **Portability**
  - Dagster and Prefect are portable and can run in multiple environments.

- **Operational ergonomics**
  - Good UIs and logging; more ML‑friendly in some respects.
  - However, you must still layer your own experiment lineage, checkpointing strategy, and sweep semantics.

- **Scaling model**
  - Executors / agents manage parallelism.
  - GPU training typically requires custom integrations similar to Airflow.

---

## Argo Workflows / Kubeflow Pipelines (brief)

- **Durability semantics**
  - Strong container‑level durability via Kubernetes; each step is a pod.

- **Long‑running processes**
  - Well‑suited to long‑running containerized tasks.

- **Code‑first expressiveness**
  - Workflows are usually specified via YAML or Python DSLs.
  - Implementing adaptive sweeps and TPE‑style logic requires more plumbing than in a code‑native workflow engine like Temporal.

- **Deterministic replay**
  - Runs are tracked in CRDs or metadata stores, but there is no built‑in, code‑level replay similar to Temporal.

- **Portability**
  - Kubernetes‑centric; portable across clusters but requires k8s expertise.

- **Operational ergonomics**
  - Strong if you already operate Kubernetes clusters; heavier‑weight for simple demos or small teams.

---

## When Temporal is the better fit

Temporal (as used in `bert_sweeps`) is particularly strong when:

- You care about **durable, resumable experiment sweeps** with checkpointed training.
- You want to **express ladder/TPE logic directly in code**, not in YAML or fragmented state machines.
- You need **replayable histories** to debug why a particular set of hyperparameters was chosen.
- You plan to **scale GPU training and CPU orchestration independently** across machines or clusters.

Other systems can run hyperparameter sweeps, but Temporal gives you:

- A first‑class notion of workflows, children, and history.
- A deterministic execution model that makes adaptive search easier to reason about.
- A clean separation between orchestration (this repo) and ML libraries (activities).
