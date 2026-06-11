---
description: Generate a complete, PR-ready cookbook recipe from a pattern description or proposal card
allowed-tools: Read, Write, Bash, Glob
argument-hint: <pattern description or proposal card>
---

# Recipe-ify

Generate a complete, runnable, PR-ready AI Cookbook recipe for the pattern in `$ARGUMENTS`
— a freeform description ("a RAG pipeline recipe using OpenAI") or a proposal card from
`/ai-cookbook:recipe-scout`.

**Audience:** AI engineers comfortable with LLMs and agents but new to Temporal. The AI
pattern is the hero; Temporal is the invisible durability layer underneath. Make the AI
concept clear and don't over-explain Temporal mechanics.

## Follow the conventions — do not restate them

Read these references first and follow them as the authoritative source. Do **not** inline
or paraphrase the conventions; they live in exactly one place:

- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/structure.md` — the canonical
  README walkthrough.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/layout.md` — files, directories,
  naming, and mandatory tests.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/frontmatter.md` — front-matter
  schema, tag accept-list, and priority bands.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/code-conventions.md` — the
  Temporal/Python rules and quality bar.

## Generate

1. Choose the category (`foundations`, `agents`, `deep_research`, `mcp`) and recipe name; the
   directory is `{category}/{recipe-name}_python`.
2. Produce **all** files per `layout.md` — runnable, not stubs: `pyproject.toml`,
   `README.md`, `worker.py`, `start_workflow.py`, `activities/`, `workflows/`, and `tests/`
   (mandatory; mock the LLM/API so the suite passes with no API key).
3. Write the README in the **canonical walkthrough shape** from `structure.md` (H1 → intro →
   `## Create the {Component}` code-sandwich sections → `## Running`). Not the brief
   overview style.
4. Apply every rule in `code-conventions.md` (`max_retries=0`, `pydantic_data_converter`,
   `start_to_close_timeout`, `ApplicationError` at boundaries, current model names, …) and
   `frontmatter.md` (schema, tag accept-list and order, priority band).

## After generating

1. Run `uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-lint recipe-lint {category}/{recipe-name}_python`
   and fix every error and warning it reports.
2. Report: the created directory; how to run it (`uv sync`, then the worker and
   `start_workflow`); any env vars needed (API keys); and any deliberate simplifications
   made to keep the recipe bite-sized.
