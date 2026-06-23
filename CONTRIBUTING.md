# Contributing to the Temporal AI Cookbook

Thanks for your interest in contributing! The cookbook is a collection of self-contained, runnable examples. Contributions can be new recipes, improvements to existing ones, bug fixes, or documentation updates.

## Prerequisites

Install commands below use Homebrew (macOS) or the official installer; pick what fits your
platform. `ruff`, `mypy`, and `pytest` need no separate install: `uv` manages them for the
recipes and the toolkit CLIs.

<!-- Recipes are Python today; add per-language prerequisites here as other languages land. -->

### Required (to run and test recipes)

| Tool | Used for | Install |
| :--- | :--- | :--- |
| Python 3.10+ | recipes are Python | `brew install python` or [python.org/downloads](https://www.python.org/downloads/) |
| [`uv`](https://docs.astral.sh/uv/) | dependency management for every recipe and the toolkit CLIs | `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `brew install uv` |
| [Temporal CLI](https://docs.temporal.io/cli) | local dev server (`temporal server start-dev`) | `brew install temporal` or `curl -sSf https://temporal.download/cli.sh \| sh` |

### Optional (the authoring toolkit)

| Tool | Used for | Install |
| :--- | :--- | :--- |
| [`just`](https://just.systems) | task runner for the toolkit check commands (`justfile`) | `brew install just` |
| [`vale`](https://vale.sh/) | prose linter for recipe READMEs | `brew install vale` |
| [Claude Code](https://www.anthropic.com/claude-code) | runs the plugin's authoring and review commands | `npm install -g @anthropic-ai/claude-code` |

The plugin, the Temporal Docs MCP server, and the Temporal Developer skill are installed per
[`cookbook-toolkit/README.md`](cookbook-toolkit/README.md): the MCP ships with the plugin, the
skill installs separately. See [Authoring tooling](#authoring-tooling) for how to use them.

## Repository Structure

Each recipe lives in its own directory under a category folder (`agents/`, `foundations/`, `deep_research/`, or `mcp/`). Every recipe must include:

```
category/recipe-name/
├── README.md           # Required, see README requirements below
├── pyproject.toml      # Project metadata and dependencies
├── worker.py           # Starts the Temporal worker
├── start_workflow.py   # Kicks off the workflow
├── activities/         # Temporal activities (external API calls, side effects)
├── workflows/          # Temporal workflow definitions
└── tests/              # Pytest-based tests
```

Look at an existing recipe (e.g., [`foundations/hello_world_openai_responses_python`](foundations/hello_world_openai_responses_python/)) as a reference when adding something new.

## Coding Conventions

Recipe code conventions, retries via Temporal (`max_retries=0` on clients), the Pydantic
data converter, activity timeouts, naming, and the ruff + strict-typing quality bar, are
documented in
[`cookbook-toolkit/skills/recipe-writing/references/code-conventions.md`](cookbook-toolkit/skills/recipe-writing/references/code-conventions.md)
and checked by `recipe-lint` (see [Authoring tooling](#authoring-tooling)). Use `uv` for
dependencies and commit lockfile changes.

The toolkit treats two Temporal resources as the canonical authority for Temporal correctness
(determinism, activity patterns, retries, cancellation, testing): the **Temporal Docs MCP
server** (`temporal-docs`, bundled in the plugin manifest, so loading the plugin configures
it) and the **Temporal Developer skill** (`temporal:temporal-developer`, a required companion
you install separately). Scout, generate, and review all consult them, and so should you while
authoring. Install steps are in [`cookbook-toolkit/README.md`](cookbook-toolkit/README.md);
both are documented at [docs.temporal.io/with-ai](https://docs.temporal.io/with-ai).

## Authoring Tooling

The `cookbook-toolkit/` directory holds an optional, local Claude Code plugin and linter that help
you write consistent recipes. The recipe conventions live in one place,
`cookbook-toolkit/skills/recipe-writing/references/` (structure, layout, front matter, code
conventions), and everything else references them.

- **Run the checks via [`just`](https://just.systems)** (the convenient entry point;
  `justfile`):

  ```bash
  just            # list commands
  just toolkit-check <category>/<recipe>_python  # recipe-lint + Vale on one recipe
  just toolkit-report                            # full report across every recipe
  just toolkit-frontmatter                       # validate front matter for all READMEs
  ```

- **Or run the tools directly.** Lint a recipe (structure, layout, naming, links,
  Temporal/Python conventions):

  ```bash
  uv run --project cookbook-toolkit/tools/recipe-lint recipe-lint <category>/<recipe>_python
  ```

  Error-severity findings (e.g. a missing `tests/`) fail CI; the rest are advisory warnings.

- **Load the plugin** for authoring commands and the reviewer agent:

  ```bash
  claude --plugin-dir <path-to-repo>/cookbook-toolkit
  ```

  Then `/ai-cookbook:new-recipe <category>/<name>` scaffolds a recipe skeleton, `/ai-cookbook:review-recipe
  <dir>` runs a full review, `/ai-cookbook:recipe-scout <repo-url>` proposes recipes from an external repo as
  cards, and `/ai-cookbook:recipe-generate <card.yaml>` builds a full recipe from a card.

  Loading the plugin also configures the bundled `temporal-docs` MCP server. Install the
  companion Temporal Developer skill separately; see
  [`cookbook-toolkit/README.md`](cookbook-toolkit/README.md) for both.

- **Prose**: a small [Vale](https://vale.sh/) ruleset (`vale --config cookbook-toolkit/.vale.ini
  <recipe>/README.md`) flags marketing / AI-giveaway language. CI runs `recipe-lint` and
  Vale advisorily via `.github/workflows/lint-recipes.yml`.

## README Requirements

Every top-level recipe README must include a front matter block as an HTML comment at the very top of the file. This is validated by CI:

```markdown
<!--
description: A one-sentence description of what the recipe demonstrates.
tags: [category, language, provider]
priority: 500
-->
```

- `description`: plain text, one sentence. **Required**: a missing `description` (or a
  missing H1, or invalid YAML) breaks the docs build and **fails CI**.
- `tags`: array, ordered `category, language, provider`, drawn from the controlled
  vocabulary in `cookbook-toolkit/skills/recipe-writing/references/tags.json` (categories `agents`,
  `foundations`, `deep_research`, `mcp`; language `python`; providers `openai`,
  `anthropic`, `litellm`). Checked as a **warning** (reported, non-blocking).
- `priority`: an integer; higher numbers appear earlier in listings. Checked as a
  **warning**.

CI validates front matter with the same YAML parser the docs site uses. Only the
docs-breakers above fail the build; tag-vocabulary, spacing, ordering, and `priority`
issues are reported as warnings. See `cookbook-toolkit/skills/recipe-writing/references/frontmatter.md`
for the full rules.

## Running Tests

### For Python Recipes
From inside a recipe directory:

```bash
uv sync                # install dependencies
uv run pytest tests/   # run the test suite
```

All tests must pass before opening a pull request.

## Pull Request Checklist

Before submitting a PR, confirm:

- [ ] README front matter is present and valid (`description`, `tags`, `priority`)
- [ ] All tests pass locally (`uv sync && uv run pytest tests/`)
- [ ] CI checks are green
- [ ] LLM client retries are disabled (let Temporal retry instead)
- [ ] New dependencies are added to `pyproject.toml`

## Questions?

- Open a [GitHub Discussion](https://github.com/temporalio/ai-cookbook/discussions) for design questions or ideas
- Join the [Temporal Community Slack](https://temporalio.slack.com) (`#topic-ai`)
