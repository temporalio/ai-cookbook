import argparse
import asyncio
import os
from collections.abc import Sequence

from temporalio.client import Client
from temporalio.worker import Worker

from activities.agent import demo_run, run_claude
from workflows.agent import ClaudeRun

TASK_QUEUE = "claude-agent-sdk-detached"


async def _run(demo: bool) -> None:
    client = await Client.connect(
        os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    )
    await Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ClaudeRun],
        activities=[demo_run if demo else run_claude],
    ).run()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    asyncio.run(_run(parser.parse_args(argv).demo))


if __name__ == "__main__":
    main()
