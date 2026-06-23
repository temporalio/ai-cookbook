# Recipe layout and naming

Every recipe is a self-contained Python project in its own directory under a category
(`agents/`, `foundations/`, `deep_research/`, `mcp/`). This reference defines the
directory layout, the required files, naming, and the slug/URL contract.

## Stable core

Every recipe has these, and `recipe-lint` treats them as required:

| Path | Purpose |
| :--- | :--- |
| `pyproject.toml` | Project metadata + dependencies (uv). |
| `README.md` | The published docs page (see `structure.md`). |
| `worker.py` | Registers workflows + activities with Temporal and runs the Worker. |
| `start_workflow.py` | Submits a workflow execution. |
| `activities/` | Activities, LLM calls, web requests, side effects. |
| `workflows/` | Workflow definitions (pure orchestration). |
| `tests/` | Test suite. **Mandatory**: see below. |

### Tests are mandatory

Every recipe must ship a `tests/` suite, even when that means mocking the LLM/API so the
tests run without credentials (no real API key). A recipe without tests is not accepted;
`recipe-lint` reports a missing or empty `tests/` as an **error** (the one CI hard-gate
in the lint layer).

## Optional directories

Add these only when the recipe needs them:

| Path | When to use |
| :--- | :--- |
| `models/` | Pydantic/dataclass request/response models shared across files. |
| `tools/` | Tool definitions + handlers for agentic/tool-calling recipes. |
| `helpers/` | Shared helpers (e.g. tool-schema generation). |
| `agents/` | Multiple agent definitions (e.g. deep research). |
| `util/` | Small utilities (e.g. HTTP error translation). |
| `codec/`, `shared/` | Payload codec + shared models (e.g. claim check). |
| `mcp_servers/` | MCP server entrypoint(s) for `mcp/` recipes. |
| `_assets/` | Images/diagrams referenced from the README. |

## `__init__.py` convention

`__init__.py` is **optional**: a directory does not need one (recipes set
`pythonpath = ["."]` in `[tool.pytest.ini_options]`, so imports work without it). But
**if an `__init__.py` exists, it must be empty.** Do not put code, re-exports, registries,
logic, in an `__init__.py`. Put that in a named module instead.

`recipe-lint` does **not** flag a missing `__init__.py`; it flags a **non-empty** one.

## Naming

| Thing | Convention | Example |
| :--- | :--- | :--- |
| Recipe directory | `{recipe-name}_python` | `tool_call_openai_python` |
| Package name (`pyproject`) | `cookbook-{recipe-name}-python` | `cookbook-tool-call-openai-python` |
| Task queue | `{recipe-name}-task-queue` | `guardrails-hard-rules-task-queue` |
| Workflow class | `PascalCase` | `ClassifyContentWorkflow` |
| Activity functions | `snake_case` | `classify`, `create` |

The directory suffix (`_python`) also drives `recipe-lint`'s language detection and will
extend to `_typescript`, `_go`, etc.

## The slug/URL contract

The recipe **directory name is its permanent public docs URL**. `bin/sync-ai-cookbook.js`
slugifies the directory name into the page path (e.g.
`foundations/hello_world_openai_responses_python` →
`docs.temporal.io/ai-cookbook/hello-world-openai-responses-python`).

**Renaming a recipe directory breaks its published URL.** Do not rename casually, a
rename requires a coordinated `SLUG_ALIASES` entry in `temporalio/documentation` and is
handled as a separate, isolated change (never folded into consistency work).

## Standardize away

- **No stray top-level entry files** that duplicate `start_workflow.py`: e.g.
  `hello_world.py`, `claude_test.py`. Use `start_workflow.py` (and `worker.py`) as the
  entrypoints.

## Verified against the corpus

The stable core holds across the recipes, and `tests/` is present in every one. One
sanctioned variation to know about:

- **MCP recipes.** `mcp/hello_world_durable_mcp_server` has no `start_workflow.py` (the MCP
  server is the entrypoint, via `mcp_servers/`) and its directory omits the `_python`
  suffix. `recipe-lint` special-cases `mcp/` recipes: it requires `mcp_servers/` (or
  equivalent) instead of `start_workflow.py`, and does not demand the `_python` suffix.
