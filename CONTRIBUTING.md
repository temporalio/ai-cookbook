# Contributing to the Temporal AI Cookbook

Thanks for your interest in contributing! The cookbook is a collection of self-contained, runnable examples. Contributions can be new recipes, improvements to existing ones, bug fixes, or documentation updates.

## Prerequisites
```markdown
<!--
For other Recipe languages: add notes here as you add other language recipes.
-->
```
### For Python Recipes
- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/) — used for dependency management in all projects
- A running Temporal server, or use the [Temporal CLI](https://docs.temporal.io/cli) (`temporal server start-dev`) for local development

## Repository Structure

Each recipe lives in its own directory under a category folder (`agents/`, `foundations/`, `deep_research/`, or `mcp/`). Every recipe must include:

```
category/recipe-name/
├── README.md           # Required — see README requirements below
├── pyproject.toml      # Project metadata and dependencies
├── worker.py           # Starts the Temporal worker
├── start_workflow.py   # Kicks off the workflow
├── activities/         # Temporal activities (external API calls, side effects)
├── workflows/          # Temporal workflow definitions
└── tests/              # Pytest-based tests
```

Look at an existing recipe (e.g., [`foundations/hello_world_openai_responses_python`](foundations/hello_world_openai_responses_python/)) as a reference when adding something new.

## Coding Conventions

- **Let Temporal handle retries.** Set `max_retries=0` on LLM client libraries and rely on Temporal's activity retry policy instead. Client-side retries interfere with durable error handling.
- **Take advantage of AI tooling for Temporal guidance and best practices.** The [Temporal Docs MCP Server](https://docs.temporal.io/with-ai) (`https://temporal.mcp.kapa.ai`) gives your AI assistant real-time access to Temporal documentation. The [Temporal Developer Skill](https://docs.temporal.io/with-ai) adds expert-level knowledge of workflow determinism, activity patterns, retry policies, and testing strategies to agents like Claude Code, Cursor, and Codex. See [docs.temporal.io/with-ai](https://docs.temporal.io/with-ai) for setup instructions.
- **Use `uv` for dependencies.** Add dependencies in `pyproject.toml` and commit any lockfile changes.

## README Requirements

Every top-level recipe README must include a front matter block as an HTML comment at the very top of the file. This is validated by CI:

```markdown
<!--
description: A one-sentence description of what the recipe demonstrates.
tags: [category, language, provider]
priority: 500
-->
```

- `description` — plain text, one sentence. **Required** — a missing `description` (or a
  missing H1, or invalid YAML) breaks the docs build and **fails CI**.
- `tags` — array, ordered `category, language, provider`, drawn from the controlled
  vocabulary in `toolkit/skills/recipe-writing/references/tags.json` (categories `agents`,
  `foundations`, `deep_research`, `mcp`; language `python`; providers `openai`,
  `anthropic`, `litellm`). Checked as a **warning** (reported, non-blocking).
- `priority` — an integer; higher numbers appear earlier in listings. Checked as a
  **warning**.

CI validates front matter with the same YAML parser the docs site uses. Only the
docs-breakers above fail the build; tag-vocabulary, spacing, ordering, and `priority`
issues are reported as warnings. See `toolkit/skills/recipe-writing/references/frontmatter.md`
for the full rules.

## Running Tests

### For Python Recipes
From inside a recipe directory:

```bash
uv sync          # install dependencies
pytest tests/    # run the test suite
```

All tests must pass before opening a pull request.

## Pull Request Checklist

Before submitting a PR, confirm:

- [ ] README front matter is present and valid (`description`, `tags`, `priority`)
- [ ] All tests pass locally (`uv sync && pytest tests/`)
- [ ] CI checks are green
- [ ] LLM client retries are disabled (let Temporal retry instead)
- [ ] New dependencies are added to `pyproject.toml`

## Questions?

- Open a [GitHub Discussion](https://github.com/temporalio/ai-cookbook/discussions) for design questions or ideas
- Join the [Temporal Community Slack](https://temporalio.slack.com) (`#topic-ai`)
