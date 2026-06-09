from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.llm_call import CallLLMRequest, call_llm


@workflow.defn
class RecipeWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return await workflow.execute_activity(
            call_llm,
            CallLLMRequest(prompt=prompt),
            start_to_close_timeout=timedelta(seconds=30),
        )
