"""Start the documentation agent workflow."""

import asyncio
from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
from rich.console import Console
from rich.panel import Panel

from workflow import DocumentationAgent

console = Console()


async def main():
    """Execute the documentation agent workflow."""
    console.print("\n[cyan]Connecting to Temporal...[/cyan]")
    client = await Client.connect(
        "localhost:7233",
        plugins=[PydanticAIPlugin()],
    )

    user_input = console.input("\n[bold yellow]Enter a question:[/bold yellow] ")

    console.print("\n[dim]Starting agent workflow...[/dim]")
    result = await client.execute_workflow(
        DocumentationAgent.run,
        user_input,
        id="docs-agent-workflow",
        task_queue="docs-agent-queue",
        id_reuse_policy=WorkflowIDReusePolicy.TERMINATE_IF_RUNNING,
    )

    console.print()
    console.print(Panel(
        result,
        title="[bold green]Result[/bold green]",
        border_style="green",
    ))
    console.print()
    console.print(f"[dim]View in Temporal UI: http://localhost:8233/namespaces/default/workflows/docs-agent-workflow[/dim]\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        exit(0)
