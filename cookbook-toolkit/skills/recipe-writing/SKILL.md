---
name: recipe-writing
description: This skill should be used when the user asks to "write a recipe", "review a recipe", "check a recipe against the cookbook style", "cookbook-ify" a pattern or project, "validate recipe front matter", or asks about "recipe conventions" for the Temporal AI Cookbook. It is the single source of truth for cookbook recipe structure, file layout, front matter, and Python code conventions.
version: 0.1.0
---

# Recipe Writing

Author and review recipes for the Temporal AI Cookbook so they are consistent,
runnable, and render correctly as published documentation pages.

A recipe is a short, self-contained, runnable example, "here's the code, here's how
it works", not a tutorial, a Validated Pattern, or a course. Each recipe `README.md`
is published verbatim as a docs page, so its structure and front matter are
user-facing.

## Read the references first

Before writing or reviewing any recipe, read the reference files. They are the
authoritative conventions. Do not rely on memory or general knowledge:

- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/scope.md`: whether a pattern is an
  AI Cookbook recipe at all. The subject-not-ingredients test (a recipe is judged by its
  headline concept, not the Temporal primitives it uses). Read this before proposing or
  rejecting a candidate.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/structure.md`: the canonical
  README "code walkthrough" shape (H1 → intro → "Create the X" sections → Running).
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/layout.md`: directory layout,
  required files, mandatory tests, naming, and the slug/URL contract.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/frontmatter.md`: README
  front-matter schema, tag accept-list, priority bands, and YAML rules.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/code-conventions.md`: the
  Temporal/recipe Python rules and the code-quality bar.

## Run the tooling

Two tools check recipes mechanically. Both are advisory. Combine their output with
judgment. This skill is the enforcer.

- Code layer: `uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-lint recipe-lint <recipe-dir>`
- Prose layer: `vale --config ${CLAUDE_PLUGIN_ROOT}/.vale.ini <file-or-dir>`

When working in the repo directly rather than through the loaded plugin, invoke the same
tools by repo path (`cookbook-toolkit/tools/recipe-lint`, `cookbook-toolkit/.vale.ini`).

## Write or review

To write a recipe: follow `structure.md` and `layout.md` for the files and shape,
`frontmatter.md` for the README front matter, and `code-conventions.md` for the Python.
Then run `recipe-lint` and Vale and resolve their findings.

To review a recipe: run `recipe-lint` and Vale first, then check what they cannot:
code-sandwich completeness, section purpose, prose clarity, and context-dependent
Temporal capitalization. Report findings grouped error / warning / suggestion.
