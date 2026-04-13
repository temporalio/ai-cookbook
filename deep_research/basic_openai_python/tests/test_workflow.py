"""Test DeepResearchWorkflow using Temporal's test framework with a mocked invoke_model activity."""

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity
from datetime import timedelta

from activities.invoke_model import InvokeModelRequest, InvokeModelResponse
from agents.shared import (
    ResearchPlan,
    ResearchAspect,
    QueryPlan,
    SearchQuery,
    SearchResult,
    ResearchReport,
)
from workflows.deep_research_workflow import DeepResearchWorkflow

TASK_QUEUE = "test-deep-research"

# -- Canned responses keyed by response_format type name --

MOCK_RESEARCH_PLAN = ResearchPlan(
    research_question="What is 2+2?",
    key_aspects=[
        ResearchAspect(
            aspect="Basic arithmetic",
            priority=5,
            description="Verify the sum of 2 and 2",
        ),
    ],
    expected_sources=["math textbooks"],
    search_strategy="Search for basic arithmetic facts",
    success_criteria=["Confirm the answer is 4"],
)

MOCK_QUERY_PLAN = QueryPlan(
    queries=[
        SearchQuery(
            query="what is 2+2",
            rationale="Direct lookup of basic arithmetic",
            expected_info_type="factual_data",
            priority=5,
        ),
    ],
)

MOCK_SEARCH_RESULT = SearchResult(
    query="what is 2+2",
    sources=["https://example.com/math"],
    key_findings="2+2 equals 4. This is a fundamental arithmetic fact.",
    relevance_score=1.0,
    citations=["Basic Mathematics, Example Publishing"],
)

MOCK_RESEARCH_REPORT = ResearchReport(
    executive_summary="The answer to 2+2 is 4.",
    detailed_analysis="Addition of two and two yields four, a foundational arithmetic fact.",
    key_findings=["2+2 equals 4"],
    confidence_assessment="High confidence",
    citations=["Basic Mathematics, Example Publishing"],
    follow_up_questions=["What is 2+3?"],
)

# Map response_format class name -> canned model_dump
_MOCK_RESPONSES: dict[str, dict] = {
    "ResearchPlan": MOCK_RESEARCH_PLAN.model_dump(),
    "QueryPlan": MOCK_QUERY_PLAN.model_dump(),
    "SearchResult": MOCK_SEARCH_RESULT.model_dump(),
    "ResearchReport": MOCK_RESEARCH_REPORT.model_dump(),
}


@activity.defn(name="invoke_model")
async def mock_invoke_model(request: InvokeModelRequest) -> InvokeModelResponse:
    """Return canned responses based on the requested response_format."""
    if request.response_format is None:
        return InvokeModelResponse(response_model="mock text", response_format=None)

    type_name = request.response_format.__name__
    if type_name not in _MOCK_RESPONSES:
        raise ValueError(f"No mock response configured for {type_name}")

    return InvokeModelResponse(
        response_model=_MOCK_RESPONSES[type_name],
        response_format=request.response_format,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_deep_research_workflow():
    """Run the full workflow with mocked invoke_model — no API key needed."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[DeepResearchWorkflow],
            activities=[mock_invoke_model],
        ):
            result = await env.client.execute_workflow(
                DeepResearchWorkflow.run,
                "What is 2+2?",
                id="test-deep-research-wf",
                task_queue=TASK_QUEUE,
                run_timeout=timedelta(seconds=30),
            )

    # Verify the formatted report contains expected content
    assert "Deep Research Report" in result
    assert "What is 2+2?" in result
    assert "The answer to 2+2 is 4." in result
    assert "2+2 equals 4" in result
    assert "Basic Mathematics, Example Publishing" in result
    assert "What is 2+3?" in result
