from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.run_agent_stage import (
        analyze_application,
        critique_analysis,
        draft_decision,
    )
    from models.flow import (
        AgentTask,
        CriticReview,
        FixedFlowInput,
        FixedFlowResult,
        HumanReviewInput,
        HumanReviewPacket,
        HumanReviewResult,
    )
    from policy import compute_metrics, hard_stop_violations, review_reasons

ACTIVITY_TIMEOUT = timedelta(seconds=45)
RETRY_POLICY = RetryPolicy(maximum_attempts=3)
MAX_REVISIONS = 2


@workflow.defn
class FixedFlowWorkflow:
    """Run a bounded, policy-constrained mortgage underwriting flow."""

    def __init__(self) -> None:
        self._review_packet: HumanReviewPacket | None = None
        self._human_review: HumanReviewResult | None = None

    @workflow.query
    def get_review_packet(self) -> HumanReviewPacket | None:
        return self._review_packet

    @workflow.signal
    async def submit_human_review(self, review: HumanReviewInput) -> None:
        self._human_review = HumanReviewResult(
            **review.model_dump(), timestamp=workflow.now().isoformat()
        )

    @workflow.run
    async def run(self, request: FixedFlowInput) -> FixedFlowResult:
        application = request.application
        metrics = compute_metrics(application)
        violations = hard_stop_violations(application, metrics)

        # @@@SNIPSTART bounded-fixed-flow
        analysis = await workflow.execute_activity(
            analyze_application,
            AgentTask(application=application, metrics=metrics, model=request.model),
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RETRY_POLICY,
        )

        critique = CriticReview(accepted=False, issues=[], revision_instructions=[])
        for revision_count in range(MAX_REVISIONS + 1):
            critique = await workflow.execute_activity(
                critique_analysis,
                AgentTask(
                    application=application,
                    metrics=metrics,
                    previous_analysis=analysis,
                    model=request.model,
                ),
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )
            if critique.accepted or revision_count == MAX_REVISIONS:
                break
            analysis = await workflow.execute_activity(
                analyze_application,
                AgentTask(
                    application=application,
                    metrics=metrics,
                    previous_analysis=analysis,
                    critique=critique,
                    model=request.model,
                ),
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )

        recommendation = await workflow.execute_activity(
            draft_decision,
            AgentTask(
                application=application,
                metrics=metrics,
                previous_analysis=analysis,
                critique=critique,
                model=request.model,
            ),
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RETRY_POLICY,
        )
        # @@@SNIPEND

        reasons = review_reasons(recommendation, critique, violations)
        human_review = None
        if reasons:
            # @@@SNIPSTART policy-human-gate
            self._review_packet = HumanReviewPacket(
                application=application,
                metrics=metrics,
                analysis=analysis,
                critique=critique,
                recommendation=recommendation,
                policy_violations=violations,
                review_reasons=reasons,
            )
            await workflow.wait_condition(lambda: self._human_review is not None)
            human_review = self._human_review
            final_decision = human_review.decision
            # @@@SNIPEND
        else:
            final_decision = recommendation.decision

        return FixedFlowResult(
            case_id=application.case_id,
            final_decision=final_decision,
            metrics=metrics,
            analysis=analysis,
            critique=critique,
            recommendation=recommendation,
            policy_violations=violations,
            revision_count=revision_count,
            human_review_required=bool(reasons),
            human_review=human_review,
        )
