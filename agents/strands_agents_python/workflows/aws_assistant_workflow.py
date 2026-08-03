from datetime import timedelta

from temporalio import workflow
from temporalio.contrib.strands import TemporalAgent, TemporalMCPClient
from temporalio.contrib.strands.workflow import activity_as_tool

# activities.tools imports only stdlib at module level (requests is imported lazily
# inside the activity), so it is safe to import directly into the workflow sandbox.
from activities.tools import get_recent_aws_announcements

SYSTEM_PROMPT = (
    "You are an AWS expert assistant. Use the AWS documentation tools to answer "
    "questions about AWS services, and the announcements tool to report recent "
    "launches. Cite documentation links when they are relevant."
)


@workflow.defn
class AWSAssistantWorkflow:
    def __init__(self) -> None:
        # Reference the MCP server registered on the worker by name. The plugin runs
        # the server's list-tools / call-tool operations as Temporal Activities.
        # cache_tools avoids re-listing the server's tools on every model turn.
        aws_docs = TemporalMCPClient(
            server="aws-docs",
            cache_tools=True,
            start_to_close_timeout=timedelta(seconds=60),
        )

        self.agent = TemporalAgent(
            start_to_close_timeout=timedelta(seconds=120),
            system_prompt=SYSTEM_PROMPT,
            tools=[
                # MCP tool: AWS documentation search/read, served over stdio.
                aws_docs,
                # Non-deterministic tool backed by a Temporal Activity (live HTTP call).
                activity_as_tool(
                    get_recent_aws_announcements,
                    start_to_close_timeout=timedelta(seconds=30),
                ),
            ],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        # TemporalAgent drives the agentic loop; every model call, tool call, and MCP
        # call runs as a durable Temporal Activity.
        result = await self.agent.invoke_async(prompt)
        return str(result)
