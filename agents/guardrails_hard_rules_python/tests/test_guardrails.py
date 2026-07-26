import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter

from models.signals import ContentSignals
from models.verdict import Verdict
from guardrails.hard_rules import _hard_block, apply_hard_rules
from activities.classify import classify, ClassifyRequest
from workflows.classify_workflow import ClassifyContentWorkflow

TASK_QUEUE = "test-guardrails-task-queue"


# --- Unit tests: hard rule logic (no Temporal, no LLM) ---

class TestHardBlock:
    def test_email_triggers_block(self):
        signals = ContentSignals(text="Email me at foo@bar.com", author_id="u1")
        result = _hard_block(signals)
        assert result is not None
        assert result.classification == "block"
        assert result.overridden_by_hard_rule is True

    def test_phone_triggers_block(self):
        signals = ContentSignals(text="Call me at 555-867-5309", author_id="u1")
        result = _hard_block(signals)
        assert result is not None
        assert result.classification == "block"

    def test_banned_keyword_triggers_block(self):
        signals = ContentSignals(text="Click here for free money!", author_id="u1")
        result = _hard_block(signals)
        assert result is not None
        assert result.classification == "block"

    def test_clean_content_returns_none(self):
        signals = ContentSignals(text="I love hiking in the mountains.", author_id="u1")
        assert _hard_block(signals) is None


class TestApplyHardRules:
    def test_overrides_llm_safe_verdict(self):
        signals = ContentSignals(text="Call me at 555-867-5309", author_id="u1")
        llm_verdict = Verdict(classification="safe", confidence=0.8, reasoning="Seems fine.")
        result = apply_hard_rules(signals, llm_verdict)
        assert result.classification == "block"
        assert result.overridden_by_hard_rule is True
        assert "Seems fine." in result.reasoning  # LLM reasoning preserved

    def test_does_not_double_override_llm_block(self):
        signals = ContentSignals(text="Call me at 555-867-5309", author_id="u1")
        llm_verdict = Verdict(classification="block", confidence=0.95, reasoning="Spam.")
        result = apply_hard_rules(signals, llm_verdict)
        assert result is llm_verdict  # unchanged

    def test_passes_through_when_no_rule_matches(self):
        signals = ContentSignals(text="Great weather today!", author_id="u1")
        llm_verdict = Verdict(classification="safe", confidence=0.95, reasoning="All good.")
        result = apply_hard_rules(signals, llm_verdict)
        assert result is llm_verdict  # unchanged


# --- Activity test: LLM + hard rules together ---

def _mock_anthropic_response(classification: str, reasoning: str):
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"classification": classification, "confidence": 0.85, "reasoning": reasoning}

    response = MagicMock()
    response.content = [tool_block]
    return response


class TestClassifyActivity:
    @pytest.mark.asyncio
    async def test_hard_rule_overrides_llm(self):
        """LLM says 'safe' but content has email → hard rule overrides to 'block'."""
        signals = ContentSignals(text="Email me at test@example.com", author_id="u1")
        request = ClassifyRequest(signals=signals)

        mock_create = AsyncMock(return_value=_mock_anthropic_response("safe", "Looks friendly."))
        with patch("activities.classify.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.create = mock_create
            result = await classify(request)

        assert result.classification == "block"
        assert result.overridden_by_hard_rule is True
        assert "Looks friendly." in result.reasoning

    @pytest.mark.asyncio
    async def test_llm_verdict_passes_through(self):
        """No hard rules fire — LLM's 'review' verdict is returned unchanged."""
        signals = ContentSignals(text="This seems a bit aggressive.", author_id="u2")
        request = ClassifyRequest(signals=signals)

        mock_create = AsyncMock(return_value=_mock_anthropic_response("review", "Borderline tone."))
        with patch("activities.classify.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.create = mock_create
            result = await classify(request)

        assert result.classification == "review"
        assert result.overridden_by_hard_rule is False


# --- Workflow integration test ---

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_workflow_returns_activity_verdict():
    """Workflow passes signals to the activity and returns its verdict."""
    signals = ContentSignals(text="Nice day!", author_id="u1")

    @activity.defn(name="classify")
    async def mock_classify(request: ClassifyRequest) -> Verdict:
        return Verdict(classification="safe", confidence=0.9, reasoning="All good.")

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ClassifyContentWorkflow],
            activities=[mock_classify],
        ):
            result = await env.client.execute_workflow(
                ClassifyContentWorkflow.run,
                signals,
                id="test-workflow-passthrough",
                task_queue=TASK_QUEUE,
            )

    assert result.classification == "safe"
    assert result.overridden_by_hard_rule is False
