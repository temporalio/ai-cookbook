"""
Agent Execution Activity — Claude Agent SDK with Temporal

This activity wraps the Claude Agent SDK (claude-agent-sdk) to provide durable,
observable agent execution inside a Temporal workflow.

Key patterns demonstrated:
1. Background heartbeats — keeps Temporal informed during long-running tool calls
2. Staleness guard — stops heartbeating when the agent appears hung
3. Response deduplication — only accumulates text from AssistantMessage events
4. Session resumption — captures session_id and supports resuming conversations
"""

import asyncio
import os
import time
from datetime import datetime, timezone

from temporalio import activity

from models import AgentInput, AgentOutput


# How often to send a heartbeat to Temporal (seconds).
HEARTBEAT_INTERVAL = 60

# If no SDK events arrive for this long, stop heartbeating and let Temporal
# kill the activity.  This prevents a truly hung agent from blocking the
# full start_to_close_timeout (30 min).
MAX_IDLE_SECONDS = 15 * 60  # 15 minutes


@activity.defn
async def execute_agent_activity(input_data: AgentInput) -> AgentOutput:
    """
    Execute a Claude agent via the Claude Agent SDK and collect results.

    The Claude Agent SDK manages the agentic loop internally — tool selection,
    execution, and multi-turn conversation are all handled by the SDK.  This
    activity simply streams events from the SDK and collects the final response.

    Heartbeat pattern:
        A background asyncio task sends heartbeats every 60 seconds, independent
        of the SDK event stream.  This is critical because the SDK may execute
        long-running tools (e.g. git clone, large file reads) that emit no events
        for extended periods.  Without background heartbeats, Temporal would kill
        the activity for missing its heartbeat_timeout.

    Staleness guard:
        If no SDK events arrive for MAX_IDLE_SECONDS (15 min), the heartbeat
        loop exits.  Temporal's heartbeat_timeout (10 min) then fires ~25 min
        after the last event, killing a truly hung agent instead of letting it
        block the full 30-minute start_to_close_timeout.

    Response deduplication:
        The Claude Agent SDK emits both StreamEvent (incremental text chunks)
        and AssistantMessage (complete text blocks).  Both contain the same text,
        so we only accumulate from AssistantMessage to avoid duplication.
    """
    # Lazy import — avoids loading the SDK at module level, which keeps the
    # Temporal worker startup fast and avoids sandbox issues.
    from claude_agent_sdk import query
    from claude_agent_sdk.types import (
        ClaudeAgentOptions, AssistantMessage, ResultMessage, SystemMessage,
    )

    activity.logger.info(
        f"Starting agent execution: prompt_length={len(input_data.prompt)}, "
        f"model={input_data.model}"
    )

    start_time = datetime.now(timezone.utc)

    try:
        # Build SDK options
        #
        # NOTE: The SDK merges os.environ with options.env ({**os.environ, **env}).
        # If the worker runs inside Claude Code, CLAUDECODE will be set, and the
        # bundled CLI binary refuses to launch ("cannot nest Claude Code sessions").
        # We override it to empty string so the subprocess doesn't see it.
        options = ClaudeAgentOptions(
            model=input_data.model,
            max_turns=input_data.max_turns,
            permission_mode=input_data.permission_mode,
            env={"CLAUDECODE": ""},
        )
        if input_data.system_prompt:
            options.system_prompt = input_data.system_prompt

        # Session resumption: if a previous session_id is provided, tell the
        # SDK to resume from that session's JSONL file on disk.
        if input_data.resume_session_id:
            options.resume = input_data.resume_session_id
            activity.logger.info(
                f"Resuming session: {input_data.resume_session_id}"
            )

        # Heartbeat state shared between the event loop and the background task.
        heartbeat_state = {
            "event_count": 0,
            "last_event_time": time.time(),
            "done": False,
        }

        async def _heartbeat_loop():
            """
            Background task that sends Temporal heartbeats at a fixed interval.

            This runs independently of the SDK event stream so that heartbeats
            continue even when the SDK is executing a long-running tool that
            produces no events.
            """
            while not heartbeat_state["done"]:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if heartbeat_state["done"]:
                    break

                idle_seconds = time.time() - heartbeat_state["last_event_time"]

                # Staleness guard: if no events for too long, the agent may be
                # stuck.  Stop heartbeating and let Temporal's heartbeat_timeout
                # kill the activity.
                if idle_seconds > MAX_IDLE_SECONDS:
                    activity.logger.warning(
                        f"No events for {idle_seconds:.0f}s — stopping heartbeat "
                        f"(agent may be stuck)"
                    )
                    break

                activity.heartbeat(
                    f"events={heartbeat_state['event_count']}, "
                    f"idle={idle_seconds:.0f}s"
                )

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

        # Collect response, events, and session ID
        response_text = ""
        total_tokens = 0
        event_count = 0
        session_id = None

        try:
            async for event in query(
                prompt=input_data.prompt,
                options=options,
            ):
                event_count += 1
                heartbeat_state["event_count"] = event_count
                heartbeat_state["last_event_time"] = time.time()

                # Capture session_id from the init SystemMessage.
                # The SDK emits a SystemMessage with subtype='init' at the
                # start of a session.  Its data dict contains the session_id
                # which can be used to resume this conversation later.
                if isinstance(event, SystemMessage):
                    subtype = getattr(event, "subtype", None)
                    data = getattr(event, "data", None)
                    if subtype == "init" and isinstance(data, dict):
                        sid = data.get("session_id")
                        if sid:
                            session_id = sid
                            activity.logger.info(f"Captured session_id: {sid}")

                # Response deduplication: The SDK emits both StreamEvent
                # (incremental chunks) and AssistantMessage (complete blocks).
                # Both contain the same text, so we ONLY accumulate from
                # AssistantMessage to avoid duplicating the response.
                if isinstance(event, AssistantMessage):
                    # AssistantMessage.content is a list of content blocks
                    for block in event.content:
                        if hasattr(block, "text"):
                            response_text += block.text

                # Capture token usage from the final result event
                if isinstance(event, ResultMessage):
                    total_tokens = getattr(event, "total_tokens", 0) or 0

        finally:
            # Always clean up the heartbeat task
            heartbeat_state["done"] = True
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()

        activity.logger.info(
            f"Agent execution completed: events={event_count}, "
            f"response_length={len(response_text)}, "
            f"processing_time={processing_time:.2f}s"
        )

        return AgentOutput(
            status="success",
            response=response_text,
            total_tokens=total_tokens,
            num_events=event_count,
            processing_time_seconds=processing_time,
            session_id=session_id,
        )

    except Exception as e:
        activity.logger.error(f"Agent execution failed: {e}", exc_info=True)

        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()

        return AgentOutput(
            status="error",
            response="",
            error_message=str(e),
            processing_time_seconds=processing_time,
        )


@activity.defn
async def log_result_activity(output: AgentOutput) -> None:
    """
    Log the agent execution result.

    In a production system, this would persist results to a database.
    Here we keep it simple for the cookbook example.
    """
    if output.status == "success":
        activity.logger.info(
            f"Agent succeeded: {len(output.response)} chars, "
            f"{output.total_tokens} tokens, "
            f"{output.processing_time_seconds:.2f}s"
        )
    else:
        activity.logger.error(f"Agent failed: {output.error_message}")
