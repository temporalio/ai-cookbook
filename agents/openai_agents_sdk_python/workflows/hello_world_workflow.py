from temporalio import workflow
from datetime import timedelta

from agents import Agent, Runner
from temporalio.contrib import openai_agents

from activities.tools import get_weather, calculate_circle_area

@workflow.defn
class HelloWorldAgent:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            name="Hello World Agent",
            instructions="You are a helpful assistant that determines what tool to use based on the user's question.",
            # Tools for the agent to use that are defined as activities
            tools=[
                openai_agents.workflow.activity_as_tool(
                    get_weather,
                    start_to_close_timeout=timedelta(seconds=10)
                ),
                openai_agents.workflow.activity_as_tool(
                    calculate_circle_area,
                    start_to_close_timeout=timedelta(seconds=10)
                )
            ]

        )

        result = await Runner.run(agent, prompt)
        return result.final_output