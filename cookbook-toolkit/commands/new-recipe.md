---
description: Scaffold a new cookbook recipe skeleton into a category directory
allowed-tools: Read, Bash, AskUserQuestion, Glob
argument-hint: <category/recipe-name>
---

# New recipe

Scaffold a new recipe skeleton with the deterministic `recipe-scaffold` tool. This is the
manual entry point for when you don't have a proposal card; it produces the same lint-clean
skeleton `/ai-cookbook:recipe-generate` starts from, but leaves the logic and prose to you.

## Instructions

1. Gather the fields:
   - If `$ARGUMENTS` is `category/recipe-name`, take the category and kebab-case name from it.
     `category` must be one of `foundations`, `agents`, `deep_research`, `mcp`.
   - Ask (AskUserQuestion) for anything missing: recipe name (kebab-case), category, LLM
     provider (`openai` / `anthropic` / `litellm` / none), human title, one-sentence
     description, and a priority integer (band per
     `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/frontmatter.md`).

2. Run the scaffolder from the repo root. It derives the directory
   `{category}/{recipe-name}_python`, package name, task queue, and provider dependency, and
   renders a lint-clean skeleton:

   ```
   uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-scaffold recipe-scaffold \
     --name {recipe-name} \
     --category {category} \
     --title "{title}" \
     --description "{description}" \
     --priority {priority} \
     --provider {provider} \
     --into .
   ```

   Omit `--provider` for a recipe with no LLM provider. Add `--force` only to overwrite an
   existing directory.

3. Report the created directory and remind the author to: fill in the Activity logic (the
   stub raises `NotImplementedError`) and the README walkthrough sections per
   `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/structure.md`, keep tests passing
   without an API key, and run
   `uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-lint recipe-lint {category}/{recipe-name}_python`
   and `uv sync` before opening a PR.
