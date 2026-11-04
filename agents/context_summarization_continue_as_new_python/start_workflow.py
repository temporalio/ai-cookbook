import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.recipe_workflow import (
    AgentInput,
    SummarizationConfig,
    SummarizingAgentWorkflow,
)


async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    result = await client.execute_workflow(
        SummarizingAgentWorkflow.run,
        AgentInput(
            prompts=[
                "I'm planning a 3-day trip to Tokyo. Suggest a theme for each day.",
                "Expand day 1 into a morning, afternoon, and evening plan.",
                "What should I pack given those plans?",
                "Summarize the whole trip in three sentences.",
            ],
            # Force a continue-as-new every two turns so the handoff is easy to
            # see in the Temporal UI. Leave this at 0 in production and let
            # Temporal decide when history is large enough.
            config=SummarizationConfig(continue_as_new_after_turns=2),
        ),
        id="context-summarization-continue-as-new-example",
        task_queue="context-summarization-continue-as-new-task-queue",
    )
    print(f"Final message: {result.final_message}")
    print(f"Turns: {result.total_turns}  Continue-as-new handoffs: {result.compactions}")


if __name__ == "__main__":
    asyncio.run(main())
