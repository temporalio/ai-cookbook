---
description: Analyze an external GitHub repo and propose AI patterns worth turning into cookbook recipes
allowed-tools: Read, Bash, WebFetch, Glob, Grep
argument-hint: <github-url>
---

# Recipe Scout

Analyze the project at `$ARGUMENTS` and produce reviewer-ready proposal cards for the parts
that would make good AI Cookbook recipes. Write no files — just structured recommendations
a reviewer who has never seen the source project can act on.

**Audience:** The cookbook targets AI engineers comfortable with LLMs and agents but new to
Temporal. Propose *AI building blocks* — how agents think, decide, call tools, and
coordinate — with Temporal as the invisible durability layer. Do NOT propose patterns that
are primarily about Temporal orchestration, distributed systems, or infrastructure; those
belong in Temporal's own docs.

## What a "good recipe" is

Read the single source of truth before judging fit, and hold candidates to it:

- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/structure.md`
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/layout.md`
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/code-conventions.md`

## Step 1 — Fetch and analyze

Fetch the repo's README, file tree (GitHub API:
`https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1`), and key source
(LLM integration, agent/tool patterns, prompt construction, workflow definitions). Look for
these **AI building-block** patterns:

- Agentic loop (LLM looped until a stop condition)
- Forced completion (`tool_choice` constrained on the final turn)
- Tool calling; parallel tool calls
- Multi-agent coordination / supervisor
- Structured output (validated against a schema)
- Human-in-the-loop (pause for a human decision)
- Streaming output; RAG; short-term / long-term memory
- Context summarization (continue-as-new)
- Guardrails; chain-of-thought / tree-of-thought
- Prompt-injection prevention; dynamic system prompts
- Cost/token tracking; multi-provider LLM abstraction

Ignore patterns primarily about Temporal internals (workflow-ID policies, heartbeats,
signal/query handlers, replay determinism) unless they're an invisible part of an AI pattern.

## Step 2 — Produce proposal cards

Rank candidates higher when they fill a **coverage-wishlist** gap not yet in the cookbook:
RAG pipeline · streaming output · short/long-term memory · context summarization
(continue-as-new) · agent supervisor / swarm · guardrails · chain/tree-of-thought ·
cost/token tracking · trigger-based AI · web crawler.

Evaluate each candidate: (1) Is it an AI building block? (2) Well-engineered, not a demo?
(3) Self-contained (~200–400 lines, standalone)? (4) Teachable (one clear concept)?
(5) Novel vs. existing recipes (check `agents/`, `foundations/`, `deep_research/`, `mcp/`)?
(6) Fills a wishlist gap?

Rank the top 2–4. For each, write a proposal card:

- **Proposed recipe:** `{category}/{recipe-name}_python`
- **One-line description:** _(the README front-matter `description`)_
- **The problem it solves:** 2–3 sentences — what goes wrong without this pattern.
- **The pattern in the source:** a 10–25 line excerpt (or pseudocode) showing it clearest.
- **How the recipe would be structured:** a brief outline that follows the canonical
  walkthrough in `structure.md` — workflow, key activity, tool/API — 5–10 bullets.
- **Closest existing recipe and what's different.**
- **Wishlist gap filled** (if any).
- **Estimated size:** rough total line count; flag anything over ~400 lines as too complex.

After the cards, add an **Excluded patterns** section: interesting candidates filtered out,
one line each.

To build a recipe from a card, pass it to `/ai-cookbook:recipe-ify`.
