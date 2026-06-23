# Cookbook toolkit

The authoring and consistency toolkit for the Temporal AI Cookbook: a Claude Code plugin
plus two `uv` CLIs. It scaffolds, generates, lints, and reviews recipes so they stay
consistent and demonstrate correct Temporal usage.

## Components

- **`recipe-writing` skill** (`skills/recipe-writing/`): the single source of truth for
  recipe structure, layout, front matter, scope, and Python/Temporal conventions. Everything
  else references it; nothing restates it.
- **Commands** (`commands/`): `recipe-scout` (repo to proposal cards), `recipe-generate`
  (card to a finished recipe), `new-recipe` (scaffold by hand), `review-recipe` (full review).
- **`recipe-reviewer` agent** (`agents/`): the judgment reviewer launched by `review-recipe`.
- **`recipe-lint`** (`tools/recipe-lint/`): structural, layout, naming, link, and code
  checks, plus `--fix` to apply ruff autofixes and formatting with the toolkit's config.
- **`recipe-scaffold`** (`tools/recipe-scaffold/`): renders a lint-clean recipe skeleton
  from a proposal card.
- **Vale ruleset** (`.vale.ini`, `styles/`): prose linting.

The pipeline is: `recipe-scout` (emit card) -> `recipe-scaffold` (deterministic skeleton) ->
`recipe-generate` (fill logic and prose) -> `recipe-lint` -> `review-recipe`.

## Using it

Install the prerequisites and plugin (below), then drive the pipeline with the slash commands:

- `/ai-cookbook:recipe-scout <repo-url>`: analyze an external repo and emit proposal cards.
- `/ai-cookbook:recipe-generate <card.yaml>`: build a full recipe from a card.
- `/ai-cookbook:new-recipe <category>/<name>`: scaffold a skeleton by hand (no card).
- `/ai-cookbook:review-recipe <dir>`: full review (lint, Vale, tests, judgment, Temporal correctness).

Proposal cards are scratch inputs, not committed artifacts: scout saves them under `cards/`
(gitignored) and generate reads from there.

This README is the plugin reference: what it is, what it needs, and how to install it. For the
end-to-end authoring workflow (prerequisites, the `just` shortcuts, CI, and the PR checklist),
see the repo's [CONTRIBUTING.md](../CONTRIBUTING.md).

## Temporal correctness comes from canonical sources

Recipe quality is correct Temporal usage, and a model's memory is not authoritative. The
toolkit treats two sources as the canonical authority for any Temporal-correctness question
(determinism, signals, child workflows, continue-as-new, cancellation, heartbeats, activity
timeout and retry semantics, replay):

- the **Temporal Developer skill** (`temporal:temporal-developer`), and
- the **Temporal Docs MCP server** (`temporal-docs`, `https://temporal.mcp.kapa.ai`).

Scout verifies a candidate's Temporal pattern against them before proposing it; generate
consults them while writing the workflow and activities; the reviewer confirms the pattern
against them before passing it. A pattern that cannot be confirmed against these sources is a
finding, not a pass. This is stated once in
`skills/recipe-writing/references/code-conventions.md` ("Canonical Temporal sources").

## Install

### The plugin

Load it for local development:

```bash
claude --plugin-dir /path/to/ai-cookbook/cookbook-toolkit
```

This registers the `recipe-writing` skill, the commands, and the `recipe-reviewer` agent, and
declares the `temporal-docs` MCP server (see below).

### The Temporal Docs MCP server (bundled)

The plugin manifest (`.claude-plugin/plugin.json`) declares the `temporal-docs` MCP server, so
loading the plugin configures it for you. To add it by hand instead (for example, to use it
outside the plugin):

```bash
claude mcp add --transport http temporal-docs https://temporal.mcp.kapa.ai
```

Or add it to a project `.mcp.json`:

```json
{
  "mcpServers": {
    "temporal-docs": { "type": "http", "url": "https://temporal.mcp.kapa.ai" }
  }
}
```

### The Temporal Developer skill (required companion)

The skill is canonical Temporal content owned by Temporal, so the toolkit references it rather
than vendoring a copy that would drift. Install it separately by following
[docs.temporal.io/with-ai](https://docs.temporal.io/with-ai). When installed it provides the
`temporal:temporal-developer` skill, which the scout, generate, and review steps invoke.

Both the MCP server and the skill are documented at
[docs.temporal.io/with-ai](https://docs.temporal.io/with-ai).

## The CLIs

The two `uv` tools run standalone, without the plugin:

```bash
uv run --project cookbook-toolkit/tools/recipe-lint recipe-lint <recipe-dir>
uv run --project cookbook-toolkit/tools/recipe-lint recipe-lint --fix <recipe-dir>
uv run --project cookbook-toolkit/tools/recipe-scaffold recipe-scaffold --card <card.yaml> --into .
```

`recipe-lint` error-severity findings fail CI; the rest are advisory warnings.
