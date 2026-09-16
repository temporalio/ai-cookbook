import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.run_agent_stage import analyze_application
from models.flow import (
    AgentTask,
    CriticReview,
    DecisionRecommendation,
    FixedFlowInput,
    HumanReviewInput,
    MortgageApplication,
    UnderwritingAnalysis,
)
from policy import compute_metrics, hard_stop_violations, review_reasons
from workflows.fixed_flow_workflow import FixedFlowWorkflow

TASK_QUEUE = "test-fixed-flow-openai"
calls: list[str] = []


def application(case_id: str = "clean", credit_score: int = 720) -> MortgageApplication:
    return MortgageApplication(
        case_id=case_id,
        credit_score=credit_score,
        monthly_income=10_000,
        monthly_debt=3_000,
        loan_amount=400_000,
        property_value=500_000,
    )


@activity.defn(name="analyze_application")
async def mock_analyze(task: AgentTask) -> UnderwritingAnalysis:
    calls.append("revise" if task.critique else "analyze")
    return UnderwritingAnalysis(
        summary="Reviewed all supplied fields.",
        risks=[],
        missing_evidence=[],
        recommended_decision="APPROVED",
    )


@activity.defn(name="critique_analysis")
async def mock_critique(task: AgentTask) -> CriticReview:
    calls.append("critique")
    accepted = task.application.case_id == "clean"
    return CriticReview(
        accepted=accepted,
        issues=[] if accepted else ["The analysis remains unsupported."],
        revision_instructions=[] if accepted else ["Support the conclusion."],
    )


@activity.defn(name="draft_decision")
async def mock_decide(task: AgentTask) -> DecisionRecommendation:
    calls.append("decide")
    return DecisionRecommendation(
        decision="APPROVED", rationale="Mock recommendation.", conditions=[]
    )


async def run_worker(
    env: WorkflowEnvironment, workflow_id: str, request: FixedFlowInput
):
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[FixedFlowWorkflow],
        activities=[mock_analyze, mock_critique, mock_decide],
    ):
        return await env.client.execute_workflow(
            FixedFlowWorkflow.run,
            request,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )


def test_policy_checks_override_an_llm_approval() -> None:
    app = application(credit_score=610)
    violations = hard_stop_violations(app, compute_metrics(app))
    reasons = review_reasons(
        DecisionRecommendation(
            decision="APPROVED", rationale="Looks fine.", conditions=[]
        ),
        CriticReview(accepted=True, issues=[], revision_instructions=[]),
        violations,
    )
    assert violations
    assert reasons == ["Binding policy checks require human review."]


@pytest.mark.asyncio
async def test_activity_uses_structured_output_and_closes_client() -> None:
    parsed = UnderwritingAnalysis(
        summary="Structured",
        risks=[],
        missing_evidence=[],
        recommended_decision="APPROVED",
    )
    response = MagicMock(output_parsed=parsed)
    client = MagicMock()
    client.responses.parse = AsyncMock(return_value=response)
    client.close = AsyncMock()

    with patch(
        "activities.run_agent_stage.AsyncOpenAI", return_value=client
    ) as client_class:
        result = await analyze_application(
            AgentTask(application=application(), metrics=compute_metrics(application()))
        )

    assert result == parsed
    client_class.assert_called_once_with(max_retries=0)
    assert (
        client.responses.parse.await_args.kwargs["text_format"] is UnderwritingAnalysis
    )
    client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_clean_case_follows_fixed_route_without_human_review() -> None:
    calls.clear()
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        result = await run_worker(
            env, "clean-fixed-flow", FixedFlowInput(application=application())
        )

    assert calls == ["analyze", "critique", "decide"]
    assert result.final_decision == "APPROVED"
    assert result.human_review_required is False


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_revision_limit_and_policy_gate_are_deterministic() -> None:
    calls.clear()
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[FixedFlowWorkflow],
            activities=[mock_analyze, mock_critique, mock_decide],
        ):
            handle = await env.client.start_workflow(
                FixedFlowWorkflow.run,
                FixedFlowInput(application=application("needs-review", 610)),
                id="bounded-fixed-flow",
                task_queue=TASK_QUEUE,
            )
            packet = None
            for _ in range(50):
                packet = await handle.query(FixedFlowWorkflow.get_review_packet)
                if packet is not None:
                    break
                await asyncio.sleep(0.05)

            assert packet is not None
            assert len(packet.review_reasons) == 2
            await handle.signal(
                FixedFlowWorkflow.submit_human_review,
                HumanReviewInput(
                    reviewer="test-reviewer",
                    decision="REJECTED",
                    notes="Policy violation confirmed.",
                ),
            )
            result = await handle.result()

    assert calls == [
        "analyze",
        "critique",
        "revise",
        "critique",
        "revise",
        "critique",
        "decide",
    ]
    assert result.revision_count == 2
    assert result.final_decision == "REJECTED"
    assert result.human_review is not None
