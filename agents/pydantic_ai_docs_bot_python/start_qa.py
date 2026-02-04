"""Interactive CLI to ask questions to the documentation bot."""

import asyncio
import sys
import uuid
from temporalio.client import Client
from workflow import QAWorkflow

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


async def start_session():
    """Start an interactive Q&A session with a new workflow."""
    with console.status("[cyan]Connecting to Temporal...", spinner="dots"):
        client = await Client.connect("localhost:7233")

    # Generate workflow ID with UUID for uniqueness
    workflow_id = f"docs-qa-{uuid.uuid4()}"

    # Start new workflow
    with console.status("[cyan]Starting Q&A session and loading docs...", spinner="dots"):
        handle = await client.start_workflow(
            QAWorkflow.run,
            id=workflow_id,
            task_queue="docs-qa-queue",
        )
        await asyncio.sleep(2)  # Wait for docs to load

    console.print(f"✓ Session ready! [dim](workflow: {workflow_id})[/dim]", style="green")

    return client, handle, workflow_id


async def ask_question(handle, question: str) -> bool:
    """Ask a question and display the answer.

    Returns:
        True if successful, False if timeout
    """
    # Send question
    await handle.signal(QAWorkflow.ask_question, question)

    # Wait for answer with spinner
    answer = None
    with console.status("[cyan]Thinking...", spinner="dots"):
        for _ in range(60):  # Try for 30 seconds
            answer = await handle.query(QAWorkflow.get_answer, question)
            if answer:
                break
            await asyncio.sleep(0.5)

    # Display answer
    if not answer:
        console.print("\n[red]✗ Timeout waiting for answer[/red]\n")
        return False

    console.print()
    if isinstance(answer, dict):
        # Display answer in a panel
        console.print(Panel(
            Markdown(answer['answer']),
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
        ))

        if answer.get('sources'):
            console.print(f"\n[dim]📚 Sources:[/dim] {', '.join(answer['sources'])}")
        if answer.get('confidence'):
            conf = answer['confidence']
            color = "green" if conf > 0.8 else "yellow" if conf > 0.6 else "red"
            console.print(f"[dim]🎯 Confidence:[/dim] [{color}]{conf:.0%}[/{color}]")
    else:
        console.print(Panel(
            answer,
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
        ))
    console.print()

    return True


async def cleanup_session(handle, workflow_id: str):
    """Gracefully stop the workflow when session ends."""
    try:
        with console.status("[dim]Ending session...", spinner="dots"):
            await handle.signal(QAWorkflow.stop)
            # Wait for workflow to complete gracefully
            await asyncio.wait_for(handle.result(), timeout=5.0)
        console.print(f"[dim]✓ Session ended (workflow: {workflow_id})[/dim]")
    except asyncio.TimeoutError:
        console.print(f"[dim]⚠ Workflow {workflow_id} did not complete in time[/dim]")
    except Exception as e:
        console.print(f"[dim]Note: Session cleanup error: {e}[/dim]")


async def interactive_mode():
    """Run interactive Q&A session."""
    console.print("\n[bold cyan]🤖 Documentation Q&A Bot[/bold cyan]")
    console.print("[dim]Ask questions about the documentation.[/dim]")
    console.print("[dim]Type 'exit' or 'quit' to end the session, Ctrl+C to abort.[/dim]\n")

    client, handle, workflow_id = await start_session()
    console.print()

    try:
        while True:
            try:
                # Get question from user
                question = Prompt.ask("[bold yellow]❓ Question[/bold yellow]").strip()

                if not question:
                    continue

                if question.lower() in ['exit', 'quit', 'q']:
                    console.print("\n[cyan]👋 Ending session...[/cyan]")
                    break

                # Ask the question
                await ask_question(handle, question)

            except KeyboardInterrupt:
                console.print("\n\n[cyan]👋 Session interrupted...[/cyan]")
                break
            except EOFError:
                console.print("\n\n[cyan]👋 Session ended...[/cyan]")
                break
    finally:
        # Always cleanup the workflow when exiting
        await cleanup_session(handle, workflow_id)


async def single_question_mode(question: str):
    """Ask a single question and exit."""
    client, handle, workflow_id = await start_session()
    console.print()

    try:
        success = await ask_question(handle, question)
        if not success:
            sys.exit(1)
    finally:
        # Cleanup the workflow after the single question
        await cleanup_session(handle, workflow_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        asyncio.run(single_question_mode(question))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())
