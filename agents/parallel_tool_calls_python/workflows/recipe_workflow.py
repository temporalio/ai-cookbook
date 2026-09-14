import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from anthropic.types import ToolUseBlock

    from activities.llm_call import CallLlmRequest, call_llm
    from tools.registry import claude_tool_definitions, get_tool

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools to answer the user. "
    "When a question needs several tools, request them all in one turn."
)

MODEL = "claude-sonnet-4-6"


@workflow.defn
class ParallelToolAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        while True:
            message = await workflow.execute_activity(
                call_llm,
                CallLlmRequest(
                    model=MODEL,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=claude_tool_definitions(),
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )

            tool_use_blocks = [b for b in message.content if b.type == "tool_use"]

            if not tool_use_blocks:
                text_blocks = [b.text for b in message.content if b.type == "text"]
                return "\n".join(text_blocks)

            # Echo the assistant turn back verbatim so the next request carries the
            # tool_use ids Claude is waiting on results for.
            messages.append({"role": "assistant", "content": _serialize(message.content)})

            # Fan out: one asyncio task per requested tool, each awaiting its own
            # Activity. asyncio.gather runs them concurrently and preserves order, so
            # the turn takes as long as the slowest tool, not the sum of all of them.
            # gather is deterministic under Temporal: the workflow always sees results
            # in the order the tasks were created, regardless of completion order.
            tool_results = await asyncio.gather(*(self._run_tool(block) for block in tool_use_blocks))

            # Claude requires exactly one tool_result per requested tool, sent back as a
            # single user turn. Order is preserved by gather.
            messages.append({"role": "user", "content": tool_results})

    async def _run_tool(self, block: "ToolUseBlock") -> dict[str, Any]:
        spec = get_tool(block.name)
        result = await workflow.execute_activity(
            spec.activity,
            spec.request_model(**block.input),
            start_to_close_timeout=timedelta(seconds=30),
        )
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(result),
        }


def _serialize(content: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return blocks
