from datetime import timedelta

from temporalio import workflow

from activities.models import LiteLLMRequest

@workflow.defn
class HelloWorld:
    @workflow.run
    async def run(self, input: str) -> str:
        messages = [
            {"role": "system", "content": "You only respond in haikus."},
            {"role": "user", "content": input},
        ]
        response = await workflow.execute_activity(
            "activities.litellm_completion.create",
            LiteLLMRequest(
                # LiteLLM allows you to switch between models easily
                # model="gpt-4o-mini",
                model="gemini-2.5-flash-lite",
                messages=messages,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )

        message = response["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))

        return message
