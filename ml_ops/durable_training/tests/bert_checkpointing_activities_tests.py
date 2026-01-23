"""Tests for BERT fine-tuning and inference activities (checkpointing package)."""

from unittest.mock import patch

import pytest
from temporalio.testing import ActivityEnvironment

from src.workflows.train_tune.bert_finetune.bert_activities import (
    BertFineTuneConfig,
    BertFineTuneRequest,
    BertFineTuneResult,
    BertInferenceRequest,
    BertInferenceResult,
    fine_tune_bert,
    run_bert_inference,
)


class TestBertActivities:
    """Test suite for BERT activities.

    These tests verify Temporal integration and delegation to the synchronous
    helpers without importing heavy ML dependencies.
    """

    @pytest.mark.asyncio
    async def test_fine_tune_bert_delegates_to_sync(self) -> None:
        """Verify that the async fine-tune activity delegates to the sync helper."""
        activity_environment = ActivityEnvironment()
        request = BertFineTuneRequest(
            run_id="test-bert-run",
            config=BertFineTuneConfig(
                model_name="bert-base-uncased",
                dataset_name="glue",
                dataset_config_name="sst2",
                num_epochs=1,
                batch_size=8,
                learning_rate=5e-5,
                max_seq_length=64,
                use_gpu=False,
            ),
        )

        expected_result = BertFineTuneResult(
            run_id=request.run_id,
            config=request.config,
            train_loss=0.42,
            eval_accuracy=0.88,
            training_time_seconds=5.0,
            num_parameters=110_000_000,
        )

        with patch(
            "src.workflows.train_tune.bert_finetune.bert_activities._fine_tune_bert_sync",
            return_value=expected_result,
        ) as mock_sync:
            result = await activity_environment.run(fine_tune_bert, request)

        assert result == expected_result
        mock_sync.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_run_bert_inference_delegates_to_sync(self) -> None:
        """Verify that the async inference activity delegates to the sync helper."""
        activity_environment = ActivityEnvironment()
        request = BertInferenceRequest(
            run_id="test-bert-run",
            texts=["hello world", "temporal workflows are great"],
            max_seq_length=64,
            use_gpu=False,
        )

        expected_result = BertInferenceResult(
            run_id=request.run_id,
            texts=list(request.texts),
            predicted_labels=[0, 1],
            confidences=[0.9, 0.8],
        )

        with patch(
            "src.workflows.train_tune.bert_finetune.bert_activities._run_bert_inference_sync",
            return_value=expected_result,
        ) as mock_sync:
            result = await activity_environment.run(run_bert_inference, request)

        assert result == expected_result
        mock_sync.assert_called_once_with(request)
