"""CLI entrypoint for running BERT hyperparameter sweeps with Temporal.

This script is intentionally small and tutorial-friendly:

- It defines a handful of sample :class:`CoordinatorWorkflowConfig` objects for
  different model/dataset combinations.
- It builds a :class:`SweepRequest` for the ladder-style
  :class:`LadderSweepWorkflow`.
- It connects to a local Temporal server, executes the workflow, and prints a
  compact summary of the best runs.
"""

import asyncio
import random
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.train_tune.bert_sweeps.custom_types import (
    BertEvalRequest,
    BertFineTuneConfig,
    CoordinatorWorkflowConfig,
    SweepRequest,
    SweepSpace,
)
from src.workflows.train_tune.bert_sweeps.workflows import (
    LadderSweepWorkflow,
)

# --------------------------------------------------------------------
# Sample Sweep Configurations
# --------------------------------------------------------------------

# Sample 1: {Model: BERT Uncased - Dataset: Glue}
config_1 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="bert-base-uncased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=8,
        learning_rate=2e-5,
        max_seq_length=128,
        use_gpu=True,
        max_train_samples=3_000,
        max_eval_samples=2_000,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=128,
        batch_size=32,
        use_gpu=True,
    ),
)

# Sample 2 {Model: BERT Cased - Dataset: Glue}
config_2 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="bert-base-cased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=10,
        batch_size=16,
        learning_rate=3e-5,
        max_seq_length=128,
        use_gpu=True,
        max_train_samples=3_000,
        max_eval_samples=2_000,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=128,
        batch_size=32,
        use_gpu=True,
    ),
)

# Sample 3 {Model: BERT Uncased - Dataset: IMDB}
seed = random.randint(0, 10000)
config_3 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="bert-base-uncased",
        dataset_name="imdb",
        dataset_config_name="plain_text",
        num_epochs=10,
        batch_size=32,
        learning_rate=2e-5,
        max_seq_length=256,
        use_gpu=True,
        max_train_samples=5_000,
        max_eval_samples=2_000,
        seed=seed,
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="imdb",
        dataset_config_name="plain_text",
        split="test",
        max_eval_samples=1_000,
        max_seq_length=256,
        batch_size=32,
        use_gpu=True,
        seed=seed,
    ),
)
# Sample 4: {Model: DistilBERT - Dataset: Glue}
seed = random.randint(0, 10000)
config_4 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="distilbert-base-uncased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=4,  # MPS-safe
        learning_rate=5e-5,
        max_seq_length=64,  # MPS-safe
        use_gpu=True,
        max_train_samples=2_000,
        max_eval_samples=1_000,
        shuffle_before_select=True,
        seed=seed,
        # optional overrides if you added them:
        # text_field=None,
        # text_pair_field=None,
        # label_field=None,
        # task_type="auto",
    ),
    evaluation_config=BertEvalRequest(
        # run_id will be filled in by the coordinator after training
        run_id=None,
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=64,
        batch_size=32,
        use_gpu=True,
        seed=seed,
        # if you changed eval to require model_uri/model_path, leave it unset here
        # and let the coordinator populate it from the training result.
        # model_path=None,
        # model_uri=None,
    ),
    dataset_snapshot=None,  # or pass a DatasetSnapshotResult if you want reproducibility
)

# Sample 5: {Model: MiniLM-L12-H384-uncased - Dataset: Glue}
seed = random.randint(0, 10000)
config_5 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="microsoft/MiniLM-L12-H384-uncased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=4,  # still MPS-safe
        learning_rate=3e-5,  # MiniLM often prefers slightly lower LR
        max_seq_length=64,  # safe starting point
        use_gpu=True,
        max_train_samples=2_000,
        max_eval_samples=1_000,
        shuffle_before_select=True,
        seed=seed,
        # Optional schema overrides (usually not needed for GLUE)
        # text_field=None,
        # text_pair_field=None,
        # label_field=None,
        # task_type="auto",
    ),
    evaluation_config=BertEvalRequest(
        # run_id filled in by coordinator
        run_id=None,
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=64,
        batch_size=32,
        use_gpu=True,
        seed=seed,
        # If your eval requires an explicit model path/URI,
        # leave it unset here and let the coordinator fill it.
        # model_path=None,
        # model_uri=None,
    ),
    dataset_snapshot=None,  # pass a snapshot if you want strict reproducibility
)
# Sample 6: {Model: DeBERTa-v3-small - Dataset: Glue}
seed = random.randint(0, 10000)
config_6 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="microsoft/deberta-v3-small",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=2,  # DeBERTa tends to be heavier on memory (safer on MPS)
        learning_rate=2e-5,  # common stable starting LR for DeBERTa fine-tuning
        max_seq_length=64,  # start safe; bump to 96/128 once stable
        use_gpu=True,
        max_train_samples=2_000,
        max_eval_samples=1_000,
        shuffle_before_select=True,
        seed=seed,
        # Optional schema overrides (usually not needed for GLUE)
        # text_field=None,
        # text_pair_field=None,
        # label_field=None,
        # task_type="auto",
    ),
    evaluation_config=BertEvalRequest(
        run_id=None,  # coordinator fills this in
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=64,
        batch_size=32,
        use_gpu=True,
        seed=seed,
        # model_path/model_uri left for coordinator to populate from training result
    ),
    dataset_snapshot=None,
)
# Sample 7: {Model: SciBERT - Dataset: SciCite}
config_7 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="allenai/scibert_scivocab_uncased",
        dataset_name="scicite",  # start with SST-2 to validate pipeline
        dataset_config_name="default",
        use_gpu=True,
        max_train_samples=2_000,
        max_eval_samples=1_000,
        shuffle_before_select=True,
    ),
    evaluation_config=BertEvalRequest(
        run_id=None,  # coordinator fills this in
        dataset_name="scicite",
        dataset_config_name="default",
        split="validation",
        max_eval_samples=1_000,
        use_gpu=True,
    ),
    dataset_snapshot=None,  # add snapshot later for reproducibility
)

# ---------------------------------------------------------------------------
# Ladder Sample Configurations
# ---------------------------------------------------------------------------

# Give this sweep a unique experiment identifier so that all per-stage run IDs
# can be grouped together in logs and under ``./bert_runs``.
ladder_id = uuid.uuid4()  # Replace with custom naming logic as desired.

# Sample 1: HPO Scaling Ladder
ladder_config_1 = SweepRequest(
    experiment_id=f"Bert-ladder-sweep-{ladder_id}",
    base=config_6,
    space=SweepSpace(
        learning_rate=(5e-5, 1e-5),
        batch_size=[
            2,
            32,
        ],  # TODO: Increase batch size in each rung of the ladder, instead of using the ladder to select batch size. Currently, the approach heavily biases towards small batch sizes.
        num_epochs=[2, 8],
        max_seq_length=[64, 256],
    ),
    num_trials=12,  # increase this for actual research, but ensure the machine has enough compute and memory or move to an autoscaling cluster
    max_concurrency=4,
    seed=random.randint(0, 10000),
)


# ------------------------------------------------------------------------------
# Starter Main Function
# ------------------------------------------------------------------------------


async def main() -> None:
    # 1. Connect to Temporal Server using the Pydantic data converter so our
    # request/response models can be passed directly as workflow arguments.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Pick the request to run. For the tutorial this is a single ladder
    # sweep, but you can easily swap in a different ``SweepRequest`` here.
    request = ladder_config_1

    # 3. Start the workflow and wait for the result. We call the ``run`` method
    # on the workflow class directly; Temporal will assign a fresh workflow
    # execution to the ID provided below.
    result = await client.execute_workflow(
        LadderSweepWorkflow.run,
        request,
        id=f"bert-ladder-{ladder_id}",
        task_queue="bert-eval-task-queue",
    )

    # 4. Print a concise, tabular summary of the winning candidate & result.
    results = result if isinstance(result, (list, tuple)) else [result]

    print("\n=== BERT evaluation summary ===")
    header = f"{'run_id':<36} {'dataset':<20} {'split':<10} {'examples':>10} {'accuracy':>9}"
    print(header)
    print("-" * len(header))

    for item in results:
        dataset = f"{item.dataset_name}/{item.dataset_config_name}"
        print(
            f"{item.run_id:<36} "
            f"{dataset:<20} "
            f"{item.split:<10} "
            f"{item.num_examples:>10} "
            f"{item.accuracy:>9.3f}",
        )

    # If the ladder workflow annotated the best result with ablation metadata,
    # print the improvement in accuracy over the ablation baseline.
    best = results[0]
    baseline = getattr(best, "baseline_accuracy", None)
    improvement = getattr(best, "improvement_vs_baseline", None)
    if baseline is not None and improvement is not None:
        print(
            "\nBest run "
            f"{best.run_id} improved accuracy by {improvement:.3f} "
            f"over the ablation baseline "
            f"(baseline={baseline:.3f}, best={best.accuracy:.3f}).",
        )


# CLI Hook
if __name__ == "__main__":
    asyncio.run(main())
