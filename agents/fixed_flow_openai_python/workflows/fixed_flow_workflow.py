from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.run_agent_stage import run_agent_stage
    from models.flow import AgentStageRequest, FixedFlowInput, FixedFlowResult


ACTIVITY_RETRY_POLICY = RetryPolicy(maximum_attempts=3)


@workflow.defn
class FixedFlowWorkflow:
    """Run specialist agents in a fixed, deterministic sequence."""

    @workflow.run
    async def run(self, request: FixedFlowInput) -> FixedFlowResult:
        # The Workflow, rather than an LLM, chooses every stage and its order.
        # @@@SNIPSTART fixed-sequence
        analysis = await workflow.execute_activity(
            run_agent_stage,
            AgentStageRequest(
                stage="analysis",
                model=request.model,
                instructions=(
                    "You are an analysis specialist. Extract the important claims, "
                    "evidence, and assumptions from the source material. Do not make "
                    "a final recommendation."
                ),
                input=f"Topic: {request.topic}\n\nSource material:\n{request.source_material}",
            ),
            start_to_close_timeout=timedelta(seconds=45),
            retry_policy=ACTIVITY_RETRY_POLICY,
        )

        critique = await workflow.execute_activity(
            run_agent_stage,
            AgentStageRequest(
                stage="critique",
                model=request.model,
                instructions=(
                    "You are a critical reviewer. Identify unsupported claims, missing "
                    "information, contradictions, and risks in the analysis. Do not "
                    "rewrite it or make the final recommendation."
                ),
                input=(
                    f"Topic: {request.topic}\n\n"
                    f"Source material:\n{request.source_material}\n\n"
                    f"Analysis to review:\n{analysis.output}"
                ),
            ),
            start_to_close_timeout=timedelta(seconds=45),
            retry_policy=ACTIVITY_RETRY_POLICY,
        )

        recommendation = await workflow.execute_activity(
            run_agent_stage,
            AgentStageRequest(
                stage="recommendation",
                model=request.model,
                instructions=(
                    "You are a decision specialist. Produce a concise recommendation "
                    "with rationale, uncertainties, and concrete next steps. Base it "
                    "only on the supplied source, analysis, and critique."
                ),
                input=(
                    f"Topic: {request.topic}\n\n"
                    f"Source material:\n{request.source_material}\n\n"
                    f"Analysis:\n{analysis.output}\n\n"
                    f"Critique:\n{critique.output}"
                ),
            ),
            start_to_close_timeout=timedelta(seconds=45),
            retry_policy=ACTIVITY_RETRY_POLICY,
        )

        return FixedFlowResult(
            analysis=analysis.output,
            critique=critique.output,
            recommendation=recommendation.output,
        )
        # @@@SNIPEND
