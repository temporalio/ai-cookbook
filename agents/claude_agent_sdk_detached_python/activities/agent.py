import asyncio
import json
import os
import sqlite3
from contextlib import suppress
from http import HTTPStatus
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)
from temporalio import activity
from temporalio.exceptions import ApplicationError

from models import RunRequest, RunResult

EFFECT_DB = Path(os.environ.get("EFFECT_DB", Path(__file__).with_name("effects.db")))


def _record_once(operation_id: str, value: str) -> bool:
    with sqlite3.connect(EFFECT_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS effects (
                operation_id TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cursor = connection.execute(
            "INSERT OR IGNORE INTO effects (operation_id, value) VALUES (?, ?)",
            (operation_id, value),
        )
        return cursor.rowcount == 1


@activity.defn(name="run_claude")
async def run_claude(request: RunRequest) -> RunResult:
    info = activity.info()

    @tool("record_once", "Record one idempotent external effect.", {"value": str})
    async def record_once(args: dict[str, Any]) -> dict[str, Any]:
        inserted = _record_once(info.workflow_id, args["value"])
        await asyncio.sleep(request.effect_delay_seconds)
        return {
            "content": [
                {
                    "type": "text",
                    "text": "recorded" if inserted else "already recorded",
                }
            ]
        }

    effects = create_sdk_mcp_server(
        name="effects",
        version="1.0.0",
        tools=[record_once],
    )
    options = ClaudeAgentOptions(
        tools=[],
        allowed_tools=["mcp__effects__record_once"],
        mcp_servers={"effects": effects},
        strict_mcp_config=True,
        permission_mode="dontAsk",
        max_turns=3,
        # Filesystem settings can load unrelated hooks and MCP servers.
        setting_sources=[],
        env={"CLAUDECODE": ""},
    )
    client = ClaudeSDKClient(options=options)

    async def heartbeat() -> None:
        while True:
            activity.heartbeat({"attempt": info.attempt})
            await asyncio.sleep(1)

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        await client.connect()
        await client.query(
            f"{request.prompt}\n"
            f"Call record_once exactly once with value {json.dumps(request.effect)}."
        )
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                if message.is_error or message.result is None:
                    error = message.result or message.subtype
                    if message.api_error_status in {
                        HTTPStatus.BAD_REQUEST,
                        HTTPStatus.UNAUTHORIZED,
                        HTTPStatus.FORBIDDEN,
                        HTTPStatus.NOT_FOUND,
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                    }:
                        raise ApplicationError(error, non_retryable=True)
                    raise RuntimeError(error)
                return RunResult(
                    response=message.result,
                    session_id=message.session_id,
                    activity_attempt=info.attempt,
                )
        raise RuntimeError("Claude Agent SDK returned no result")
    except asyncio.CancelledError:
        with suppress(Exception):
            await asyncio.shield(client.interrupt())
        raise
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
        await client.disconnect()


@activity.defn(name="run_claude")
async def demo_run(request: RunRequest) -> RunResult:
    info = activity.info()
    inserted = _record_once(info.workflow_id, request.effect)
    print(f"attempt={info.attempt} effect_inserted={inserted}", flush=True)

    remaining = request.effect_delay_seconds
    while remaining > 0:
        activity.heartbeat({"attempt": info.attempt})
        sleep_seconds = min(remaining, 1)
        await asyncio.sleep(sleep_seconds)
        remaining -= sleep_seconds

    return RunResult(
        response="demo complete",
        session_id="demo-session",
        activity_attempt=info.attempt,
    )
