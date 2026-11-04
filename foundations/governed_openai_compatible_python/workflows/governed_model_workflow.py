from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.openai_compatible import ModelRequest, invoke_model


@workflow.defn
class GovernedModelWorkflow:
    @workflow.run
    async def run(self, request: dict[str, str]) -> str:
        workflow_id = workflow.info().workflow_id
        return await workflow.execute_activity(
            invoke_model,
            ModelRequest(
                model=request.get("model", "gpt-4o"),
                instructions="Answer concisely and accurately.",
                input=request["prompt"],
                run_id=workflow_id,
                request_id=f"req_{workflow_id}_model",
            ),
            start_to_close_timeout=timedelta(seconds=45),
        )
