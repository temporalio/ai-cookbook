import argparse
import asyncio
import json
import os
import uuid
from collections.abc import Sequence

from temporalio.client import Client

from models import RunRequest
from workflows.agent import ClaudeRun

TASK_QUEUE = "claude-agent-sdk-detached"


async def _client() -> Client:
    return await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))


async def _run(args: argparse.Namespace) -> None:
    client = await _client()
    handle = client.get_workflow_handle_for(ClaudeRun.run, args.workflow_id)
    if args.command == "submit":
        handle = await client.start_workflow(
            ClaudeRun.run,
            RunRequest(
                prompt=args.prompt,
                effect=args.effect,
                effect_delay_seconds=args.delay,
            ),
            id=args.workflow_id,
            task_queue=TASK_QUEUE,
        )
        print(json.dumps({"workflow_id": handle.id}))
    elif args.command == "status":
        description = await handle.describe()
        print(json.dumps({"workflow_id": handle.id, "status": description.status.name}))
    elif args.command == "result":
        print(json.dumps((await handle.result()).__dict__))
    elif args.command == "cancel":
        await handle.cancel(reason="cancelled from CLI")
        print(json.dumps({"workflow_id": handle.id, "status": "CANCEL_REQUESTED"}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("prompt")
    submit.add_argument("--effect", default="demo")
    submit.add_argument("--delay", type=float, default=0)
    submit.add_argument("--workflow-id", default=f"claude-run-{uuid.uuid4()}")

    for command in ("status", "result", "cancel"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("workflow_id")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(_run(_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
