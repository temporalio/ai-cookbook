---
description: Review a cookbook recipe against the conventions (recipe-lint + Vale + tests + reviewer agent)
allowed-tools: Read, Grep, Glob, Bash, Task
argument-hint: <recipe-dir>
---

# Review recipe

Review a single Temporal AI Cookbook recipe against the cookbook conventions and produce
one prioritized report.

## Instructions

1. Resolve the target recipe directory:
   - If `$ARGUMENTS` is provided, use it as the recipe directory.
   - If empty, list the recipe directories under `agents/`, `foundations/`,
     `deep_research/`, and `mcp/` and ask the user which to review.

2. Launch the **recipe-reviewer** agent (via the Task tool) against that directory. The
   agent reads the references in `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/`,
   runs `recipe-lint` and `vale`, runs the recipe's tests, applies the judgment-only
   checks, and verifies Temporal correctness against the canonical Temporal sources (the
   `temporal:temporal-developer` skill and the `temporal-docs` MCP). Wait for its report.

3. Combine the agent's findings into one report:

   ```
   ## Recipe review: <recipe-dir>

   ### Summary
   - Errors: <n>   Warnings: <n>   Suggestions: <n>

   ### Findings
   [grouped error → warning → suggestion, each with file:line and a fix]

   ### Recommended actions
   [prioritized list, errors first]
   ```

4. State plainly whether the recipe is ready to merge. Tooling is advisory; the review
   verdict is yours.
