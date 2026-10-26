from datetime import timedelta

from agents import Agent, Runner
from temporalio import workflow
from temporalio.contrib import openai_agents
from temporalio.contrib.workflow_streams import WorkflowStream

from activities.tools import calculate_circle_area, get_weather


@workflow.defn
class StreamingAgent:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        # @workflow.init's parameters must match @workflow.run's (Temporal
        # requires this); prompt itself isn't needed until run() below.
        # Hosts the model-events stream: WorkflowStream registers the
        # signal/update/query handlers that the streaming model activity
        # (configured via ModelActivityParameters.streaming_topic) publishes
        # into, and that external subscribers poll via WorkflowStreamClient.
        self._stream = WorkflowStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            name="Streaming Hello World Agent",
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

        result = Runner.run_streamed(agent, prompt)
        async for _ in result.stream_events():
            # Raw model-stream events (text deltas, tool-call argument
            # deltas) are already published by the streaming model activity
            # to the "model-events" topic. This loop just drives the run to
            # completion; external subscribers see events via subscribe.py.
            pass
        return result.final_output
