---
description: Scaffold a new cookbook recipe from the template into a category directory
allowed-tools: Read, Write, Bash, AskUserQuestion, Glob
argument-hint: <category/recipe-name>
---

# New recipe

Scaffold a new recipe by copying the skeleton template and filling its placeholders.

## Instructions

1. Determine the target:
   - If `$ARGUMENTS` is `category/recipe-name`, use it. `category` must be one of
     `foundations`, `agents`, `deep_research`, `mcp`.
   - Otherwise ask (AskUserQuestion) for: recipe name (kebab-case), category, language
     (default `python`), and LLM provider (`openai` / `anthropic` / `litellm` / none).
   - The recipe directory is `{category}/{recipe-name}_python` at the repo root.

2. Copy the skeleton from `${CLAUDE_PLUGIN_ROOT}/templates/recipe-skeleton/` into the
   target directory (preserve the tree: `activities/`, `workflows/`, `tests/`, etc.).

3. Replace the ALL_CAPS placeholders in the copied files:
   - `RECIPE_SLUG` → the kebab-case recipe name (drives the package name
     `cookbook-RECIPE_SLUG-python` and task queue `RECIPE_SLUG-task-queue`).
   - `RECIPE_TITLE` → the human title (the README H1).
   - `RECIPE_DESCRIPTION` → one plain sentence for the front-matter `description`.
   - `RECIPE_CATEGORY` → the category; `RECIPE_PROVIDER` → the provider tag (drop it from
     `tags` if none); `RECIPE_PRIORITY` → an integer in the right band (see
     `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/frontmatter.md`).

4. Remind the author to: fill in the recipe's real logic and the README walkthrough
   sections, keep tests passing without an API key, and run
   `uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-lint recipe-lint {category}/{recipe-name}_python`
   and `uv sync` before opening a PR.
