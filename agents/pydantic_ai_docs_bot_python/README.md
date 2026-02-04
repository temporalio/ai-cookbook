<!--
description: Multi-agent documentation Q&A bot using PydanticAI and Temporal
tags: [agents, python, pydantic-ai]
priority: 775
-->

# Multi-Agent Documentation Q&A Bot

This example demonstrates a multi-agent system that answers questions about technical documentation using PydanticAI's `TemporalAgent` for durable AI execution. The system uses a dispatcher-specialist pattern where a fast routing agent decides whether to answer directly or consult a documentation specialist.

PydanticAI agents are wrapped with `TemporalAgent` to make them compatible with Temporal's replay mechanism. The dispatcher agent returns a union type (`DirectAnswer | SearchDocs`) that determines the execution path.

The workflow inherits from `PydanticAIWorkflow` and uses signals to receive questions, allowing it to process multiple questions over its lifetime while maintaining durable state.

This recipe highlights these key design decisions:

- **TemporalAgent for durable AI execution**: PydanticAI's `TemporalAgent` wrapper makes AI agent calls compatible with Temporal's replay mechanism. Agent invocations are automatically converted to activities.
- **Multi-agent dispatcher pattern**: A cheap, fast model routes questions to appropriate handlers. Only complex questions requiring documentation search invoke the more capable (and expensive) model.
- **Structured outputs with PydanticAI**: Using Pydantic models and union types for type-safe AI responses. The dispatcher returns `DirectAnswer | SearchDocs`, making the routing decision explicit and type-safe.
- **Signal-driven workflow with graceful completion**: Questions are sent via Temporal signals, allowing the workflow to process multiple questions over its lifetime. A stop signal enables graceful workflow completion instead of forceful termination.
- **Interactive session lifecycle**: Each CLI session starts its own workflow instance and completes it gracefully on exit, allowing multiple parallel sessions.
- **AI provider flexibility**: Auto-detects available API keys (Anthropic, OpenAI, Google) and selects appropriate models at runtime.

## Prerequisites

- Python 3.11+
- Temporal dev server running ([install guide](https://docs.temporal.io/cli/#install))
- API key for one of: OpenAI, Anthropic, or Google

## Pydantic Models for Structured Outputs

We define Pydantic models for the dispatcher's decision types and the documentation search result.

`agents.py`
```python
from pydantic import BaseModel
from typing import Literal

class DirectAnswer(BaseModel):
    """Answer simple questions directly without searching docs."""
    type: Literal["direct"] = "direct"
    answer: str
    confidence: float

class SearchDocs(BaseModel):
    """Search documentation for detailed answer."""
    type: Literal["search"] = "search"
    query: str
    keywords: list[str]

# Union type for dispatcher decisions
DispatcherDecision = DirectAnswer | SearchDocs

class DocAnswer(BaseModel):
    """Structured answer from documentation search."""
    answer: str
    sources: list[str]
    confidence: float
```

## PydanticAI Agents with Auto-Detection

Agents are configured to auto-detect which AI provider is available.

`agents.py`
```python
from pydantic_ai import Agent

# Auto-detect available API provider
def _select_models() -> tuple[str, str]:
    """Select models based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return ("anthropic:claude-haiku-4-5", "anthropic:claude-sonnet-4-5")
    elif os.getenv("OPENAI_API_KEY"):
        return ("openai:gpt-4o-mini", "openai:gpt-4o")
    elif os.getenv("GOOGLE_API_KEY"):
        return ("google:gemini-2.0-flash-exp", "google:gemini-2.0-flash-exp")
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")

_dispatcher_model, _docs_model = _select_models()

# Dispatcher Agent - routes questions using fast, cheap model
dispatcher_agent = Agent(
    _dispatcher_model,
    output_type=DispatcherDecision,
    name='dispatcher',  # Required for Temporal
    system_prompt="""You are a question routing agent for a documentation Q&A system.

Your job is to decide how to handle each question:

1. DirectAnswer: Use for simple, general questions that don't need documentation.
   - Example: "What is Temporal?" → Direct answer
   - Confidence should be 0.8+ if you're sure

2. SearchDocs: Use for questions requiring specific documentation details.
   - Example: "How do I retry an activity?" → Search docs
   - Extract 2-5 keywords for search

Be conservative: when in doubt, search docs for accurate answers.""",
)

# Documentation Agent - searches docs using more capable model
docs_agent = Agent(
    _docs_model,
    output_type=DocAnswer,
    name='docs_search',  # Required for Temporal
    system_prompt="""You are a documentation search agent.

Given documentation chunks and a search query, your job is to:
1. Analyze the documentation for relevant information
2. Formulate a clear, concise answer
3. List which documentation sources you used
4. Provide a confidence score (0.7-1.0)

Be direct and practical in your answers.""",
)
```

## Wrapping Agents with TemporalAgent

PydanticAI agents must be wrapped with `TemporalAgent` to work with Temporal's replay mechanism.

`workflow.py`
```python
from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow, TemporalAgent
from agents import dispatcher_agent, docs_agent

# Wrap agents for Temporal compatibility
temporal_dispatcher = TemporalAgent(dispatcher_agent)
temporal_docs_agent = TemporalAgent(docs_agent)
```

## Workflow with PydanticAIWorkflow

The workflow inherits from `PydanticAIWorkflow` and registers the temporal agents.

`workflow.py`
```python
@workflow.defn
class QAWorkflow(PydanticAIWorkflow):
    """Q&A workflow that routes questions through multi-agent system."""

    # Register temporal agents
    __pydantic_ai_agents__ = [temporal_dispatcher, temporal_docs_agent]

    def __init__(self):
        self._questions = asyncio.Queue()
        self._answers: dict[str, str | dict] = {}
        self._docs: dict[str, str] = {}
        self._should_stop = False

    @workflow.run
    async def run(self) -> str:
        """Run the Q&A workflow."""
        # Load documentation at startup
        self._docs = await workflow.execute_activity(
            load_docs,
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Process questions as they arrive
        while not self._should_stop:
            await workflow.wait_condition(
                lambda: not self._questions.empty() or self._should_stop
            )

            if self._should_stop:
                break

            question_text = await self._questions.get()

            try:
                answer = await self._process_question(question_text)
                self._answers[question_text] = answer
            except Exception as e:
                self._answers[question_text] = f"Error: {str(e)}"

        return "Q&A session complete"

    @workflow.signal
    async def ask_question(self, question: str):
        """Signal to ask a question."""
        await self._questions.put(question)

    @workflow.signal
    def stop(self):
        """Signal to gracefully stop the workflow."""
        self._should_stop = True

    @workflow.query
    def get_answer(self, question: str) -> str | dict | None:
        """Query to get an answer."""
        return self._answers.get(question)
```

## Workflow Lifecycle: Graceful Completion

Unlike simpler workflows that use `execute_workflow()` and complete after a single operation, this recipe demonstrates a **long-running interactive workflow** that processes multiple questions over its lifetime.

### Why Use This Pattern?

This pattern is appropriate when:
- You need to maintain state across multiple interactions (loaded documentation, conversation history)
- Setup cost is high (loading docs from disk/network)
- Multiple operations share context

For single-shot operations, the simpler `execute_workflow()` pattern (see [Basic Agentic Loop](../agentic_loop_tool_call_openai_python/)) is more appropriate.

### Graceful Completion with Stop Signal

The workflow uses a stop signal for clean shutdown instead of forceful termination:

`start_qa.py`
```python
async def cleanup_session(handle, workflow_id: str):
    """Gracefully stop the workflow when session ends."""
    await handle.signal(QAWorkflow.stop)
    # Wait for workflow to complete naturally
    await asyncio.wait_for(handle.result(), timeout=5.0)
```

**Why not terminate()?**
- `terminate()` forcefully kills the workflow without allowing cleanup
- Stop signal allows the workflow to break out of its loop and return normally
- The workflow completes with status "Completed" instead of "Terminated"
- This is the recommended pattern for long-running workflows that need graceful shutdown

### Session Management

Each CLI invocation creates its own workflow instance:

```python
workflow_id = f"docs-qa-{uuid.uuid4()}"
handle = await client.start_workflow(
    QAWorkflow.run,
    id=workflow_id,
    task_queue="docs-qa-queue",
)
```

This allows multiple parallel CLI sessions with independent workflow instances.

## Processing Questions with TemporalAgent

The `TemporalAgent` automatically handles converting agent calls to activities.

`workflow.py`
```python
    async def _process_question(self, question: str) -> str | dict:
        """Process a question through the multi-agent system."""
        # Route through dispatcher (automatically becomes an activity)
        dispatcher_result = await temporal_dispatcher.run(question)
        decision = dispatcher_result.output

        # Handle decision
        if isinstance(decision, DirectAnswer):
            return decision.answer

        elif isinstance(decision, SearchDocs):
            # Search documentation
            relevant_docs = {}
            for title, content in self._docs.items():
                if any(keyword.lower() in content.lower() for keyword in decision.keywords):
                    relevant_docs[title] = content

            if not relevant_docs:
                relevant_docs = self._docs

            # Format docs for the agent
            docs_context = "\n\n---\n\n".join([
                f"[{title}]\n{content[:500]}..."
                for title, content in list(relevant_docs.items())[:3]
            ])

            prompt = f"""Query: {decision.query}
Keywords: {', '.join(decision.keywords)}

Documentation:
{docs_context}

Based on these docs, provide a structured answer."""

            # Call docs agent (automatically becomes an activity)
            docs_result = await temporal_docs_agent.run(prompt)
            doc_answer = docs_result.output

            return {
                "answer": doc_answer.answer,
                "sources": doc_answer.sources,
                "confidence": doc_answer.confidence,
            }

        return "Unable to process question"
```

## Worker with PydanticAIPlugin

The worker must include `PydanticAIPlugin` to handle TemporalAgent activities.

`worker.py`
```python
from temporalio.client import Client
from temporalio.worker import Worker
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from workflow import QAWorkflow, load_docs

async def main():
    """Start the Temporal worker."""
    # Connect to Temporal with PydanticAI plugin
    client = await Client.connect(
        "localhost:7233",
        plugins=[PydanticAIPlugin()],
    )

    # Create and run worker
    worker = Worker(
        client,
        task_queue="docs-qa-queue",
        workflows=[QAWorkflow],
        activities=[load_docs],  # TemporalAgent activities are auto-registered
    )

    print("Worker started. Waiting for questions...")
    await worker.run()
```

## Running

### Start Temporal Dev Server

```bash
temporal server start-dev
```

### Install Dependencies

```bash
cd agents/pydantic_ai_docs_bot_python
uv sync
```

### Add Documentation

Create a `docs/` directory and add markdown files:

```bash
mkdir docs
curl -o docs/workflows.md https://raw.githubusercontent.com/temporalio/documentation/main/docs/develop/python/core-application.md
```

You can add any markdown documentation files to this directory.

### Configure API Key

Copy the example and add your API key:

```bash
cp .env.example .env
# Edit .env and uncomment one of:
#   OPENAI_API_KEY=sk-your-key-here
#   ANTHROPIC_API_KEY=sk-ant-your-key-here
#   GOOGLE_API_KEY=your-key-here
```

### Start the Worker

```bash
uv run python worker.py
```

Wait for "Worker started. Waiting for questions..." message.

### Ask Questions

In a new terminal, you can use either interactive or single-question mode.

Each CLI session starts its own workflow and completes it gracefully when you exit. You can run multiple CLI instances simultaneously for parallel sessions.

**Interactive Mode** (recommended):
```bash
uv run python start_qa.py

# Then ask multiple questions in the session:
❓ Question: What is Temporal?
# ... answer ...
❓ Question: How do I write a workflow?
# ... answer ...
❓ Question: exit
# Workflow completes gracefully
```

**Single Question Mode**:
```bash
uv run python start_qa.py "What is Temporal?"
# Workflow starts, answers, and completes gracefully
```

## Related Recipes

- [Tool Calling Agent](../tool_call_openai_python/) - Foundation for basic tool calling patterns
- [Basic Agentic Loop](../agentic_loop_tool_call_openai_python/) - More complex agent orchestration

## Resources

- [PydanticAI Documentation](https://ai.pydantic.dev/) - Official docs for PydanticAI framework
- [PydanticAI with Temporal](https://ai.pydantic.dev/agents/#temporalagent) - TemporalAgent integration guide
- [VibeCheck: Building a production AI agent system with Pydantic](https://www.youtube.com/watch?v=3rpwaKQXI7A) - Live coding session showcasing native Pydantic Temporal support. 