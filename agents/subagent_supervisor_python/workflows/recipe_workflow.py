from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.exceptions import ChildWorkflowError

with workflow.unsafe.imports_passed_through():
    from anthropic.types import Message

    from activities.llm_call import CallLlmRequest, call_llm
    from models.schemas import SubagentRequest, SubagentResult
    from tools.registry import (
        DELEGATE_TO_SUBAGENT,
        SUBAGENT_TOOLS,
        delegate_tool_schema,
        subagent_tool_schemas,
    )

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 10

SUPERVISOR_SYSTEM = (
    "You are a supervisor agent. For self-contained sub-tasks, call delegate_to_subagent "
    "with a focused task and the tool names the subagent needs. Use the subagent's result "
    "to answer the user."
)
SUBAGENT_SYSTEM = "You are a focused subagent. Use your tools to complete the task, then reply with the answer."


def _assistant_content(message: Message) -> list[dict[str, Any]]:
    # Convert Claude's content blocks back into the dict shape the Messages API expects
    # when we append the assistant turn to the running conversation.
    content: list[dict[str, Any]] = []
    for block in message.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
    return content


def _final_text(message: Message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


@workflow.defn
class SubagentWorkflow:
    @workflow.run
    async def run(self, request: SubagentRequest) -> SubagentResult:
        # A small bounded agent loop with only the granted tools. The delegate tool is
        # never in this set, so the subagent cannot spawn further subagents.
        messages: list[dict[str, Any]] = [{"role": "user", "content": request.task}]
        tools = subagent_tool_schemas(request.tool_names)

        for _ in range(MAX_TURNS):
            message = await workflow.execute_activity(
                call_llm,
                CallLlmRequest(model=MODEL, system=SUBAGENT_SYSTEM, messages=messages, tools=tools),
                start_to_close_timeout=timedelta(seconds=30),
            )
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if not tool_uses:
                return SubagentResult(result=_final_text(message))

            messages.append({"role": "assistant", "content": _assistant_content(message)})
            results: list[dict[str, Any]] = []
            for block in tool_uses:
                handler = SUBAGENT_TOOLS[block.name]
                output = handler(block.input["text"])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            messages.append({"role": "user", "content": results})

        return SubagentResult(result="Subagent stopped: turn limit reached.")


@workflow.defn
class SupervisorAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        # The supervisor's agentic loop. Its only tool is delegate_to_subagent; calling it
        # runs a child workflow and the result comes back as a tool_result.
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tools = [delegate_tool_schema()]

        for _ in range(MAX_TURNS):
            message = await workflow.execute_activity(
                call_llm,
                CallLlmRequest(model=MODEL, system=SUPERVISOR_SYSTEM, messages=messages, tools=tools),
                start_to_close_timeout=timedelta(seconds=30),
            )
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if not tool_uses:
                return _final_text(message)

            messages.append({"role": "assistant", "content": _assistant_content(message)})
            results: list[dict[str, Any]] = []
            for block in tool_uses:
                output = await self._delegate(block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            messages.append({"role": "user", "content": results})

        return "Supervisor stopped: turn limit reached."

    async def _delegate(self, tool_input: dict[str, Any]) -> str:
        # Strip the delegate tool from the granted set so the subagent can't recurse,
        # and derive a deterministic child id from the parent id plus a workflow-safe uuid.
        granted = [n for n in tool_input.get("tool_names", []) if n != DELEGATE_TO_SUBAGENT]
        child_id = f"{workflow.info().workflow_id}-subagent-{workflow.uuid4()}"
        try:
            result = await workflow.execute_child_workflow(
                SubagentWorkflow.run,
                SubagentRequest(task=tool_input["task"], tool_names=granted),
                id=child_id,
                task_queue=workflow.info().task_queue,
                # No RetryPolicy on the child: workflows don't retry by default, and we don't
                # want one here. Retries live in the subagent's call_llm Activity, where they belong.
                static_summary="delegate_to_subagent:run",
            )
        except ChildWorkflowError as exc:
            # A failed subagent surfaces back to the model as a tool error instead of crashing
            # the supervisor, so the supervisor can react: retry, delegate differently, or give up.
            # The child's underlying failure (e.g. an exhausted-retry Activity error) is in exc.cause.
            return f"Subagent failed: {exc.cause or exc}"
        return result.result
