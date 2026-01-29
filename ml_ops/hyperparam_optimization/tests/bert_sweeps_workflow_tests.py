"""Tests for the BERT sweep workflows (bert_sweeps package).

These tests focus on orchestration behavior and determinism. They replace
heavy ML activities with lightweight mocks so that we can exercise
`CheckpointedBertTrainingWorkflow`, `BertEvalWorkflow`, `CoordinatorWorkflow`,
and `LadderSweepWorkflow` end-to-end inside the Temporal test environment.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from src.workflows.train_tune.bert_sweeps.custom_types import (
    BertEvalRequest,
    BertEvalResult,
    BertFineTuneConfig,
    BertFineTuneRequest,
    BertFineTuneResult,
    CoordinatorWorkflowConfig,
    DatasetSnapshotRequest,
    DatasetSnapshotResult,
    SweepRequest,
    SweepSpace,
)
from src.workflows.train_tune.bert_sweeps.workflows import (
    BertEvalWorkflow,
    CheckpointedBertTrainingWorkflow,
    CoordinatorWorkflow,
    LadderSweepWorkflow,
)


class TestBertSweepsWorkflows:
    """Test suite for BERT sweep workflows using mocked activities."""

    @pytest.fixture
    def task_queue(self) -> str:
        """Generate a unique task queue name for each test."""
        return f"test-bert-sweeps-{uuid.uuid4()}"

    @pytest.mark.asyncio
    async def test_ladder_sweep_workflow_with_mocked_activities(
        self,
        client: Client,
        task_queue: str,
    ) -> None:
        """Run LadderSweepWorkflow end-to-end with lightweight mocks."""

        @activity.defn(name="create_dataset_snapshot")
        async def create_dataset_snapshot_mocked(
            request: DatasetSnapshotRequest,
        ) -> DatasetSnapshotResult:
            """Mocked snapshot activity that returns a tiny, deterministic snapshot."""
            return DatasetSnapshotResult(
                snapshot_id=f"{request.dataset_name}-{request.dataset_config}-test",
                dataset_name=request.dataset_name,
                dataset_config=request.dataset_config,
                num_train_samples=request.max_samples or 100,
                num_eval_samples=0,
                data_hash="deadbeefdeadbeef",
                snapshot_timestamp="2024-01-01T00:00:00Z",
                snapshot_path="./data_snapshots/test-snapshot",
            )

        @activity.defn(name="fine_tune_bert")
        async def fine_tune_bert_mocked(
            request: BertFineTuneRequest,
        ) -> BertFineTuneResult:
            """Mocked fine-tuning activity for testing."""
            return BertFineTuneResult(
                run_id=request.run_id,
                config=request.config,
                train_loss=0.5,
                eval_metrics={"accuracy": 0.8},
                training_time_seconds=1.0,
                num_parameters=110_000_000,
                dataset_snapshot=request.dataset_snapshot,
                total_checkpoints_saved=1,
                inferred_text_field="text",
                inferred_text_pair_field=None,
                inferred_label_field="label",
                inferred_task_type="classification",
                inferred_num_labels=2,
            )

        @activity.defn(name="evaluate_bert_model")
        async def evaluate_bert_model_mocked(
            request: BertEvalRequest,
        ) -> BertEvalResult:
            """Mocked evaluation activity for testing."""
            return BertEvalResult(
                run_id=request.run_id or "test-bert-run",
                dataset_name=request.dataset_name,
                dataset_config_name=request.dataset_config_name,
                split=request.split,
                num_examples=request.max_eval_samples or 100,
                accuracy=0.9,
            )

        @activity.defn(name="set_seed")
        async def set_seed_mocked(seed: int) -> int:
            """Mocked seed-jitter activity used by ladder sweeps."""
            return seed

        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[
                CheckpointedBertTrainingWorkflow,
                BertEvalWorkflow,
                CoordinatorWorkflow,
                LadderSweepWorkflow,
            ],
            activities=[
                create_dataset_snapshot_mocked,
                fine_tune_bert_mocked,
                evaluate_bert_model_mocked,
                set_seed_mocked,
            ],
            activity_executor=ThreadPoolExecutor(5),
        ):
            base_config = CoordinatorWorkflowConfig(
                fine_tune_config=BertFineTuneConfig(
                    model_name="bert-base-uncased",
                    dataset_name="glue",
                    dataset_config_name="sst2",
                    num_epochs=1,
                    batch_size=4,
                    learning_rate=3e-5,
                    max_seq_length=64,
                    use_gpu=False,
                    max_train_samples=200,
                    max_eval_samples=100,
                    shuffle_before_select=True,
                    seed=123,
                ),
                evaluation_config=BertEvalRequest(
                    dataset_name="glue",
                    dataset_config_name="sst2",
                    split="validation",
                    max_eval_samples=100,
                    max_seq_length=64,
                    batch_size=16,
                    use_gpu=False,
                    model_path="./bert_runs/test-bert-sweeps",
                    seed=123,
                ),
            )

            sweep_request = SweepRequest(
                experiment_id="test-bert-sweeps",
                base=base_config,
                space=SweepSpace(
                    learning_rate=(1e-5, 5e-5),
                    batch_size=[2, 4],
                    max_seq_length=[64, 128],
                    num_epochs=[1, 2],
                ),
                num_trials=4,
                max_concurrency=2,
                seed=42,
            )

            results = await client.execute_workflow(
                LadderSweepWorkflow.run,
                sweep_request,
                id=f"test-bert-ladder-{uuid.uuid4()}",
                task_queue=task_queue,
            )

            # LadderSweepWorkflow returns a leaderboard of BertEvalResult entries.
            assert isinstance(results, list)
            assert results
            for item in results:
                assert isinstance(item, BertEvalResult)
                assert item.dataset_name == "glue"
                assert item.dataset_config_name == "sst2"
                assert item.split == "validation"
                assert item.num_examples > 0
                assert 0.0 <= item.accuracy <= 1.0
