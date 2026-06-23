---
description: Generate a complete, PR-ready cookbook recipe from a proposal card
allowed-tools: Read, Write, Bash, Glob
argument-hint: <path to a proposal-card YAML file>
---

# Recipe-generate

Turn the proposal card at `$ARGUMENTS` into a complete, runnable, PR-ready AI Cookbook
recipe. The card is a YAML file in the format from
`${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/proposal-card.md` — usually emitted
by `/ai-cookbook:recipe-scout`, or hand-authored against that format.

This command does **not** invent structure. `recipe-scaffold` renders the deterministic
skeleton from the card's `recipe:` block; your job is to fill the stubs with real Activity
logic and README prose using the card's `context:` block.

**Audience:** AI engineers comfortable with LLMs and agents but new to Temporal. The AI
pattern is the hero; Temporal is the invisible durability layer underneath. Make the AI
concept clear and don't over-explain Temporal mechanics.

## Step 1 — Scaffold from the card

Run the deterministic scaffolder. It validates the card against `card-schema.json`, fails
fast on a malformed card, and renders a lint-clean skeleton at `{category}/{name}_python`:

```
uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-scaffold \
  recipe-scaffold --card $ARGUMENTS --into .
```

If it errors, the card is invalid — fix the card and rerun. Do not hand-create the directory.

After it runs you have all the files from `layout.md` in place: `pyproject.toml`, `README.md`,
`worker.py`, `start_workflow.py`, `activities/llm_call.py` (a stub raising
`NotImplementedError`), `workflows/recipe_workflow.py`, and `tests/test_workflow.py`. The
front matter, package name, task queue, and provider dependency are already correct.

## Step 2 — Follow the conventions — do not restate them

Read these references and follow them as the authoritative source. Do **not** inline or
paraphrase the conventions:

- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/structure.md` — the canonical
  README walkthrough.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/layout.md` — files, directories,
  naming, mandatory tests.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/code-conventions.md` — the
  Temporal/Python rules and quality bar.

## Step 3 — Fill the stubs

Using the card's `context:` block (`problem`, `source_excerpt`, `structure_outline`,
`closest_recipe`, and any `notes`) as the design input:

1. Replace the `NotImplementedError` Activity stub with the real LLM call and pattern logic,
   applying every rule in `code-conventions.md` (`max_retries=0` on the client constructor
   only, `pydantic_data_converter`, `start_to_close_timeout`, `ApplicationError` at
   boundaries, current model names, …).
2. Flesh out the workflow to orchestrate the pattern from `structure_outline`.
3. Make the tests pass with no API key (mock the LLM/API). Keep the mandatory `tests/`.
4. Rewrite `README.md` into the **canonical walkthrough shape** from `structure.md` (H1 →
   intro → `## Create the {Component}` code-sandwich sections → `## Running`), using the
   card's `problem` for the intro. Leave the front matter the scaffolder produced intact.

Do not edit the deterministic surface the scaffolder owns (package name, task queue,
directory name, front-matter tags) unless the card was wrong — in which case fix the card and
rerun Step 1.

## Step 4 — Verify and report

1. Run `uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-lint recipe-lint {category}/{name}_python`
   and fix every error and warning it reports.
2. Run `uv sync && uv run pytest tests/` in the recipe directory and confirm the suite passes.
3. Report: the created directory; how to run it (`uv sync`, then the worker and
   `start_workflow`); any env vars needed (API keys); and any deliberate simplifications made
   to keep the recipe bite-sized.
