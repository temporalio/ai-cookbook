# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A collection of self-contained, runnable examples (called "recipes") for building durable AI agents and workflows with [Temporal](https://temporal.io/). Each recipe lives in its own directory under a category folder and can be run independently.

Categories: `foundations/` (core patterns), `agents/` (agentic loops and tool calling), `deep_research/` (multi-agent research), `mcp/` (durable MCP servers).

## Commands

All Python recipes use [`uv`](https://docs.astral.sh/uv/) for dependency management. Commands are run from inside a recipe directory:

```bash
uv sync                  # install/sync dependencies
uv run pytest tests/     # run the full test suite
uv run python -m worker  # start the Temporal worker (separate terminal)
uv run python -m start_workflow  # trigger a workflow execution
```

To run a single test:
```bash
uv run pytest tests/test_foo.py::test_bar -v
```

CI runs tests with a 30-second timeout per test (`pytest tests/ --timeout=30`).

## Recipe Structure

Every recipe follows this layout:

```
category/recipe-name/
├── README.md           # Must include front matter (see CI Requirements)
├── pyproject.toml      # Dependencies and pytest config
├── worker.py           # Registers workflows/activities with Temporal
├── start_workflow.py   # Submits a workflow execution
├── workflows/          # Temporal workflow definitions
├── activities/         # Temporal activities (LLM calls, external I/O)
└── tests/              # pytest-based tests
```

## Architecture Patterns

**Temporal handles durability; activities handle external calls.** Workflows orchestrate the sequence; activities are where LLM API calls, web requests, and other side effects happen. This separation means workflows survive crashes and can retry failed activities without re-running successful ones.

**Retries belong to Temporal, not the LLM client.** All LLM clients are configured with `max_retries=0`. Activity retry policies in the workflow definition control retry behavior instead. Client-side retries interfere with Temporal's durable error handling.

**Pydantic models are used for serialization.** Recipes use `pydantic_data_converter` (from `temporalio.contrib.pydantic`) to serialize Pydantic v2 models as workflow/activity inputs and outputs. Request/response dataclasses or Pydantic models define the data contract between orchestrator and activities.

**Activities are kept generic where possible.** Rather than hardcoding model names or system prompts into activities, recipes pass them as parameters so workflows control behavior without re-registering activities.

LLM providers used across recipes: OpenAI (Messages API, Responses API), Anthropic/Claude, LiteLLM (provider-agnostic), OpenAI Agents SDK.

## CI Requirements

**README front matter** — Every top-level recipe README must start with this HTML comment block (validated by CI):

```markdown
<!--
description: A one-sentence description of what the recipe demonstrates.
tags: [category, language, provider]
priority: 500
-->
```

The validator parses with the same YAML library the docs site uses and tiers its findings:

- **Hard errors (fail CI — they break the docs build):** missing `description`, missing
  H1 title, or invalid YAML.
- **Warnings (reported, non-blocking):** `tags` not in the controlled vocabulary
  (`toolkit/skills/recipe-writing/references/tags.json`), `tags:[` spacing, wrong tag order
  (`category, language, provider`), missing/non-integer `priority`, or forbidden
  `last_updated`/`title` keys.

Consistency is enforced by the `recipe-writing` skill and the `recipe-lint` tool under
`toolkit/`, not by hard CI gates. See `toolkit/skills/recipe-writing/references/` for the
full conventions.

**Tests must pass** before a PR is merged. CI detects changed recipe directories and runs `uv sync && pytest tests/ --timeout=30` for each.

## Local Development Prerequisites

- Python 3.10+
- `uv` installed
- A running Temporal server: `temporal server start-dev` (via [Temporal CLI](https://docs.temporal.io/cli))

## Temporal Guidance

The [Temporal Docs MCP Server](https://docs.temporal.io/with-ai) (`https://temporal.mcp.kapa.ai`) provides real-time access to Temporal documentation and is useful when working on workflow determinism, activity patterns, retry policies, and testing strategies.
