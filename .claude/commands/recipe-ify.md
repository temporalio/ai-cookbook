# Recipe-ify

Analyze an external project and generate a draft recipe (or multiple) for the ai-cookbook.

Usage: `/project:recipe-ify <github-url>`

## What you do

You are an expert at extracting teachable, self-contained AI patterns from real-world projects and turning them into cookbook recipes in the exact format this repo uses.

**Audience reminder:** The AI Cookbook targets AI Engineers who are comfortable with LLMs and agents but are new to Temporal. Recipes should teach *AI building blocks* — patterns for how agents think, decide, call tools, and coordinate — with Temporal providing durability underneath. Do NOT extract patterns that are primarily about Temporal orchestration, distributed systems, or infrastructure; those belong in Temporal's own documentation, not here.

Work through the following steps.

---

### Step 1 — Fetch and analyze the project

Fetch the repository at `$ARGUMENTS`. Collect:
- The README (for intent and architecture overview)
- The full file tree (via GitHub API: `https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1`)
- Key source files: LLM integration code, agent/tool patterns, prompt construction, workflow definitions

Look specifically for these **AI building block** patterns, which make strong recipes:
- **Agentic loop** — LLM called in a loop until a stop condition (tool use, stop sequence, empty tool calls)
- **Forced completion** — On the final loop iteration, `tool_choice` is constrained to a specific tool so the agent must commit to a decision rather than looping forever
- **Tool calling** — LLM invokes structured tools; results fed back into the conversation
- **Parallel tool calls** — LLM requests multiple tools simultaneously; all results must be collected before the next turn
- **Multi-agent coordination / agent supervisor** — One agent spawns or delegates to sub-agents; results are aggregated
- **Structured output** — LLM output is parsed and validated against a Pydantic schema
- **Human-in-the-loop** — Workflow pauses and waits for a human decision before continuing
- **Streaming output** — Activity emits incremental tokens/chunks rather than waiting for full completion
- **RAG (retrieval-augmented generation)** — Retrieved context injected into the prompt before calling the LLM
- **Short-term memory** — Conversation history carried across turns within a single workflow run
- **Long-term memory** — Facts or summaries persisted across workflow runs and retrieved on demand
- **Context summarization** — Long conversation history compressed (e.g., via `continue_as_new`) to stay within context limits
- **Guardrails** — LLM output checked against a policy before being acted on; rejected outputs are blocked or re-requested
- **Chain-of-thought / tree-of-thought** — LLM explicitly reasons through steps before producing a final answer
- **Prompt injection prevention** — Untrusted external data is isolated from control instructions (e.g., XML tags, separate message turns)
- **Dynamic system prompts** — System instructions constructed at runtime from context (user prefs, retrieved docs, current state)
- **Cost/token tracking** — Token usage recorded per workflow run for budgeting or rate-limiting
- **Multi-provider LLM abstraction** — Single interface that dispatches to Anthropic, OpenAI, LiteLLM, or local models

Ignore patterns that are primarily about Temporal internals (workflow ID policies, heartbeats, signal/query handlers, replay determinism) unless they are a natural, invisible part of an AI pattern above.

---

### Step 2 — Recommend which patterns to extract

The cookbook has a wishlist of use cases not yet covered. Patterns that fill one of these gaps should be ranked higher:
- RAG pipeline
- Streaming output
- Short-term or long-term memory
- Context summarization (ContinueAsNew)
- Agent supervisor / multi-agent swarm
- Guardrails
- Chain-of-thought / tree-of-thought
- Cost/token tracking
- Trigger-based AI (event-driven or timer-based)
- Web crawler

For each candidate pattern you find, evaluate:
1. **Is it an AI building block?** Would an AI engineer immediately recognize this as a useful pattern for their LLM/agent work, independent of what orchestrator they use?
2. **Is it well-engineered, not a demo?** The cookbook publishes reference-quality code, not flashy one-offs. Does the pattern reflect how a real production system would be built?
3. **Is it self-contained?** Can it stand alone as a 200–400 line recipe without pulling in the entire project?
4. **Is it teachable?** Does it demonstrate a single clear concept a developer can learn from?
5. **Is it novel vs. existing recipes?** Check existing recipes in this repo (foundations/, agents/, deep_research/, mcp/) and flag if a very similar recipe already exists.
6. **Does it fill a wishlist gap?** Cross-reference against the coverage wishlist above.

Rank the top 2–4 patterns. For each, provide a proposal card with the following sections — written so a reviewer who has never seen the source project can evaluate it:

**Proposed recipe:** `{category}/{recipe-name}_python`

**One-line description:** _(the README front matter `description` field)_

**The problem it solves:** In 2–3 sentences: what goes wrong if a developer doesn't know this pattern? What mistake do they typically make, and what does that cost them?

**The pattern in the source:** A short code excerpt (10–25 lines) from the source project that shows the pattern at its clearest. If the source isn't Python or doesn't translate directly, describe the equivalent pseudocode. This is the "exhibit A" that justifies the recipe.

**How the recipe would be structured:** A brief outline — what the workflow does, what the key activity does, what tool or API is involved. Not full code, just enough for a reviewer to picture the finished recipe. Should be 5–10 bullet points.

**Closest existing recipe and what's different:** Name the most similar recipe already in the cookbook and state specifically what this recipe adds or changes. If there's no close match, say so.

**Wishlist gap filled:** Which item from the coverage wishlist does this address, if any? If none, say so.

**Estimated size:** Rough line count for the finished recipe (all files combined). Flag anything over 400 lines as potentially too complex for a single recipe.

If a pattern you found is interesting but primarily a Temporal/infrastructure concern rather than an AI building block, list it in a separate "Excluded patterns" section with a one-line reason for each.

Ask the user which recipe(s) to generate before proceeding.

---

### Step 3 — Generate the recipe scaffold

For each approved recipe, generate ALL of the following files. Use the exact conventions from CLAUDE.md and the existing recipes in this repo. Generate complete, runnable files — not stubs.

**File conventions:**
- Directory: `{category}/{recipe-name}_python/`
- Task queue: `{recipe-name}-task-queue`
- Workflow class: `PascalCaseWorkflow`
- Activity functions: `snake_case`
- Request models: `ActivityNameRequest` (dataclass or Pydantic BaseModel)
- LLM clients: always `max_retries=0`
- Timeouts: `start_to_close_timeout=timedelta(seconds=30)` (adjust up for research tasks)
- Data converter: `pydantic_data_converter` everywhere (Client.connect, WorkflowEnvironment)
- Python version constraint: `>=3.10,<3.14`
- Temporalio: `>=1.15.0,<2`

**Files to generate:**

#### `README.md`
Must start with this exact front matter block:
```
<!--
description: One-sentence description of what the recipe demonstrates.
tags: [category, python, provider]
priority: 500
-->
```
Then: title, brief overview, what it demonstrates, how to run (uv sync, uv run python -m worker, uv run python -m start_workflow), what to expect in output.

#### `pyproject.toml`
- `name = "cookbook-{recipe-name}-python"`
- `requires-python = ">=3.10,<3.14"`
- `dependencies`: temporalio + the relevant LLM provider SDK
- `[dependency-groups] dev`: pytest, pytest-timeout, pytest-asyncio
- `[tool.pytest.ini_options] pythonpath = ["."]`

#### `worker.py`
- `Client.connect("localhost:7233", data_converter=pydantic_data_converter)`
- `Worker(client, task_queue=..., workflows=[...], activities=[...])`
- `asyncio.run(main())`

#### `start_workflow.py`
- Connect, execute_workflow, print result

#### `workflows/{workflow_name}.py`
- `@workflow.defn` class
- `@workflow.run` async method
- Activities called via `workflow.execute_activity`
- Always specify `start_to_close_timeout`

#### `activities/{activity_name}.py`
- `@activity.defn` functions
- LLM client with `max_retries=0`
- Non-retryable errors raise `ApplicationError(..., non_retryable=True)`
- Request model at top of file or in `activities/models.py`

#### `tests/test_{recipe_name}.py`
- `@pytest.mark.asyncio` + `@pytest.mark.timeout(30)` on all tests
- `WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)`
- Mock activities using `unittest.mock.AsyncMock` or by injecting test doubles into the Worker
- At minimum: one happy-path workflow test

---

### Step 4 — Write the files

Create the directory and write all files. Then report:
- What was created
- How to run it: `cd {directory} && uv sync && uv run pytest tests/`
- Any simplifications you made vs. the source project, and why
- What a developer should modify to connect to real APIs (env vars needed, etc.)
