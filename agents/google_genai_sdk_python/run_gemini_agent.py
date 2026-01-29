"""
Simple launcher for the Gemini agent.

Usage examples:

- Start the worker (in one terminal):
  python run_gemini_agent.py worker

- Run a query (in another terminal):
  python run_gemini_agent.py run "Where am I?"

Environment:
- Requires a Temporal server at localhost:7233
- Requires GOOGLE_API_KEY set with Gemini API access

This script reuses the existing agent code in agents/google_genai_sdk_python/.
"""

import asyncio
import os
import sys
import uuid
import argparse


def _ensure_module_path() -> None:
    """Ensure the Gemini agent package directory is importable.

    The existing files use top-level imports (e.g., `from workflows.agent import ...`).
    We add that directory to sys.path so those imports resolve correctly.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    agent_dir = os.path.join(repo_root, "agents", "google_genai_sdk_python")
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)


def _require_env(var_name: str) -> None:
    if not os.environ.get(var_name):
        print(f"Environment variable {var_name} is required but not set.")
        sys.exit(1)


async def run_worker() -> None:
    """Start the Temporal worker that runs the Gemini agent workflow and activities."""
    # Defer imports until sys.path is prepared
    from temporalio.client import Client
    from temporalio.worker import Worker
    from temporalio.contrib.pydantic import pydantic_data_converter

    from workflows.agent import AgentGeminiWorkflow
    from activities import gemini_responses, tool_invoker
    from concurrent.futures import ThreadPoolExecutor

    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="tool-invoking-agent-gemini-task-queue",
        workflows=[AgentGeminiWorkflow],
        activities=[
            gemini_responses.create,
            tool_invoker.dynamic_tool_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=10),
    )
    print("Worker started on task queue: tool-invoking-agent-gemini-task-queue")
    await worker.run()


async def run_query(prompt: str) -> None:
    """Execute a single workflow run for the provided user prompt."""
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    from workflows.agent import AgentGeminiWorkflow

    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    result = await client.execute_workflow(
        AgentGeminiWorkflow.run,
        prompt,
        id=f"agentic-loop-id-{uuid.uuid4()}",
        task_queue="tool-invoking-agent-gemini-task-queue",
    )
    print(f"Result: {result}")


def main() -> None:
    _ensure_module_path()

    parser = argparse.ArgumentParser(description="Run the Gemini agent worker or a single query.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    sp_worker = subparsers.add_parser("worker", help="Start the Temporal worker")
    sp_worker.set_defaults(func=lambda args: run_worker())

    sp_run = subparsers.add_parser("run", help="Run a single query through the agent")
    sp_run.add_argument("prompt", nargs="?", default="Tell me about recursion")
    sp_run.set_defaults(func=lambda args: run_query(args.prompt))

    args = parser.parse_args()

    # Gemini API key is required for tool-calling responses
    _require_env("GOOGLE_API_KEY")

    # Run the selected command
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
