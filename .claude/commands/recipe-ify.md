# Recipe-ify

Generate a complete, PR-ready recipe for the AI Cookbook from a pattern description.

Usage: `/project:recipe-ify <pattern description or proposal card>`

The input can be:
- A proposal card from `/project:recipe-scout`
- A freeform description ("generate a RAG pipeline recipe using OpenAI")
- Anything in between

---

## What you do

You are an expert at writing reference-quality AI Cookbook recipes. Generate ALL files for the recipe described in `$ARGUMENTS`, following the conventions below exactly. Produce complete, runnable files — not stubs or placeholders.

**Audience reminder:** Recipes target AI Engineers who are comfortable with LLMs and agents but are new to Temporal. The AI pattern is the hero; Temporal is the invisible durability layer underneath. Don't over-explain Temporal mechanics — focus on making the AI concept clear.

---

## Cookbook conventions

**Directory:** `{category}/{recipe-name}_python/`
Categories: `foundations` (single LLM call or simple pattern), `agents` (agentic loops, tool use), `deep_research` (multi-agent), `mcp` (MCP servers)

**Naming:**
- Task queue: `{recipe-name}-task-queue`
- Workflow class: `PascalCaseWorkflow`
- Activity functions: `snake_case`
- Request/response models: `ActivityNameRequest`, `ActivityNameResponse`

**Always:**
- LLM clients: `max_retries=0` — Temporal handles retries, not the client
- Data converter: `pydantic_data_converter` everywhere — in `Client.connect()`, in `WorkflowEnvironment.start_time_skipping()`
- Activity timeouts: always specify `start_to_close_timeout` (30s default; increase for research/LLM tasks)
- Non-retryable errors: catch them and raise `ApplicationError(..., non_retryable=True)`
- Python: `>=3.10,<3.14`
- Temporalio: `>=1.15.0,<2`

---

## Files to generate

### `README.md`

Must open with this exact front matter block:
```
<!--
description: One-sentence description of what the recipe demonstrates.
tags: [category, python, provider]
priority: 500
-->
```
Then: title, 1–2 paragraph overview of what the recipe teaches, prerequisites, how to run:
```
uv sync
uv run python -m worker        # terminal 1
uv run python -m start_workflow  # terminal 2
```
End with what to expect in the output.

### `pyproject.toml`
```toml
[project]
name = "cookbook-{recipe-name}-python"
version = "0.1"
description = "..."
authors = [{ name = "Temporal Technologies Inc", email = "sdk@temporal.io" }]
requires-python = ">=3.10,<3.14"
readme = "README.md"
license = "MIT"
dependencies = [
    "temporalio>=1.15.0,<2",
    # LLM provider SDK
]

[dependency-groups]
dev = [
    "pytest>=9.0.3",
    "pytest-timeout>=2.4.0",
    "pytest-asyncio>=0.26.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
```

### `worker.py`
```python
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter

async def main():
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(client, task_queue="...-task-queue", workflows=[...], activities=[...])
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### `start_workflow.py`
Connect with `pydantic_data_converter`, call `execute_workflow`, print result.

### `workflows/{name}.py`
- `@workflow.defn` class with `@workflow.run` method
- Calls activities via `workflow.execute_activity(fn, request, start_to_close_timeout=...)`
- Pure orchestration — no LLM calls, no I/O

### `activities/{name}.py`
- `@activity.defn` functions
- LLM client initialized with `max_retries=0`
- Request model defined at top of file (dataclass or Pydantic `BaseModel`)
- Catch non-retryable API errors → `ApplicationError(..., non_retryable=True)`

### `tests/test_{name}.py`
- `@pytest.mark.asyncio` and `@pytest.mark.timeout(30)` on every test
- Use `WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)`
- Register mock activities in the Worker to avoid real API calls
- Cover at minimum: happy path, and the key edge case the recipe is about

---

## After generating files

1. Report what was created and the directory path
2. Show how to run: `cd {dir} && uv sync && uv run pytest tests/`
3. List any env vars needed (API keys, etc.) and where to set them
4. Note any deliberate simplifications made to keep the recipe bite-sized
