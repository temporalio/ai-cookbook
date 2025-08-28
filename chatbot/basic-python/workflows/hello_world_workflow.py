from temporalio import workflow
from datetime import timedelta

from activities import openai_responses


@workflow.defn
class HelloWorldAgent:
    @workflow.run
    async def run(self, input: str) -> str:
        system_instructions = "You only respond in haikus."
        result = await workflow.execute_activity(
            openai_responses.create,
            openai_responses.OpenAIResponsesRequest(
                instructions=system_instructions,
                input=input,
            ),
            schedule_to_close_timeout=timedelta(seconds=30),
        )
        return result
 