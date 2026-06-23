# ABOUTME: Task runner for the cookbook-toolkit deterministic checks. Run from the repo root.
# The AI tools (recipe-writing skill, recipe-reviewer agent, /ai-cookbook:* commands) run
# inside a Claude Code session and are not invocable from here.

# List available commands.
default:
    @just --list

# Lint one recipe (structure, layout, naming, links, Temporal/Python conventions).
toolkit-lint recipe:
    uv run --project cookbook-toolkit/tools/recipe-lint recipe-lint {{recipe}}

# Check one recipe's README prose with Vale.
toolkit-vale recipe:
    vale --config cookbook-toolkit/.vale.ini {{recipe}}/README.md

# Run both deterministic checks on one recipe.
toolkit-check recipe: (toolkit-lint recipe) (toolkit-vale recipe)

# Validate front matter across all recipe READMEs.
toolkit-frontmatter:
    node .github/scripts/validate-frontmatter.js

# Full consistency report across every recipe (recipe-lint + Vale + front matter).
toolkit-report:
    #!/usr/bin/env bash
    set -uo pipefail
    echo "# Cookbook consistency report"
    echo
    echo "## Front matter"
    node .github/scripts/validate-frontmatter.js || true
    echo
    fail=0
    for recipe in agents/*/ foundations/*/ deep_research/*/ mcp/*/; do
        [ -f "$recipe/pyproject.toml" ] || continue
        recipe="${recipe%/}"
        echo "## $recipe"
        uv run --project cookbook-toolkit/tools/recipe-lint recipe-lint "$recipe" || fail=1
        vale --config cookbook-toolkit/.vale.ini "$recipe/README.md" || true
        echo
    done
    if [ "$fail" -ne 0 ]; then
        echo "One or more recipes have error-severity findings."
        exit 1
    fi
    echo "No error-severity findings across the corpus."
