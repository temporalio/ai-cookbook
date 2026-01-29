from temporalio import workflow
from datetime import timedelta

from activities import openai_responses


@workflow.defn
class HelloWorld:
    @workflow.run
    async def run(self, input: str) -> str:
        system_instructions = "You only respond in haikus."
        result = await workflow.execute_activity(
            openai_responses.create,
            openai_responses.OpenAIResponsesRequest(
                model="gpt-4o-mini",
                instructions=system_instructions,
                input=input,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )
        return result.output_text
