---
description: Analyze an external GitHub repo and propose AI patterns worth turning into cookbook recipes
allowed-tools: Read, Bash, WebFetch, Glob, Grep
argument-hint: <github-url>
---

# Recipe Scout

Analyze the project at `$ARGUMENTS` and produce machine-ingestible proposal cards for the
parts that would make good AI Cookbook recipes. Each card is a YAML file `recipe-generate`
can feed straight into `recipe-scaffold`: and a reviewer who has never seen the source
project can still read.

**Audience:** The cookbook targets AI engineers comfortable with LLMs and agents but new to
Temporal. Propose *AI building blocks* (how agents think, decide, call tools, coordinate,
remember, or recover) built durably on Temporal. Showing a known agent pattern implemented
in Temporal is exactly the goal.

The exclusion is narrow and about a recipe's **subject, not its ingredients**: leave out
patterns whose headline concept *is* a Temporal mechanic with no agent concept on top
(workflow-ID policies, heartbeat tuning, replay determinism). A recipe is free to use any
Temporal primitive it needs (continue-as-new, child workflows, signals, timers) when an AI
building block is the headline and the primitive is invisible plumbing. Do not reject a
candidate just because it uses signals or child workflows. Read
`${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/scope.md` (the SSOT for the in/out
test) before filtering anything out.

## What a "good recipe" is

Read the single source of truth before judging fit, and hold candidates to it:

- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/scope.md`: is this an AI Cookbook
  recipe at all (the subject-not-ingredients test, with worked examples).
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/structure.md`
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/layout.md`
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/code-conventions.md`

The card you emit has a fixed shape. Read it before Step 2:

- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/proposal-card.md`: the card's two
  blocks (`recipe:` deterministic, `context:` prose) and a worked example.
- `${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/card-schema.json`: the
  authoritative schema `recipe-scaffold` validates against.

## Step 1: Fetch and analyze

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

Ignore a pattern only when its *subject* is a Temporal mechanic with no agent concept on top
(workflow-ID policies, heartbeat tuning, replay determinism). A pattern that *uses* signals,
child workflows, or continue-as-new to serve an AI building block is in scope; judge by the
subject, not the primitives. See `scope.md`.

When a candidate's value rests on a Temporal pattern (determinism, child workflows, signals,
continue-as-new, cancellation, heartbeats), verify it against the canonical Temporal sources
before you propose it: invoke the `temporal:temporal-developer` skill and query the
`temporal-docs` MCP. The card's `structure_outline`, `source_excerpt`, and `notes` must
describe correct Temporal usage, since `recipe-generate` builds the recipe from them. See
`code-conventions.md` ("Canonical Temporal sources").

## Step 2: Produce proposal cards

Rank candidates higher when they fill a **coverage-wishlist** gap not yet in the cookbook:
RAG pipeline · streaming output · short/long-term memory · context summarization
(continue-as-new) · agent supervisor / swarm · guardrails · chain/tree-of-thought ·
cost/token tracking · trigger-based AI · web crawler.

Evaluate each candidate: (1) Is it an AI building block? (apply the subject-not-ingredients
test from `scope.md`; don't reject on Temporal primitives alone) (2) Well-engineered, not a demo?
(3) Self-contained (~200-400 lines, standalone)? (4) Teachable (one clear concept)?
(5) Novel vs. existing recipes (check `agents/`, `foundations/`, `deep_research/`, `mcp/`)?
(6) Fills a wishlist gap?

Rank the top 2-4. For each, emit one proposal card as a fenced ```yaml block in the exact
format from `proposal-card.md`, valid against `card-schema.json`:

- `recipe:`: the deterministic fields. `name` (kebab-case), `category` (`agents` /
  `foundations` / `deep_research` / `mcp`), `language: python`, `provider` (0+ of
  `openai`/`anthropic`/`litellm`; first drives the client), `title` (README H1),
  `description` (one sentence), `priority` (band per `frontmatter.md`), and optional
  `components` (`workflow_class`, `activities[]`, `tools[]`). These map straight onto the
  directory, package, and task queue, pick `name` carefully.
- `context:`: the reviewer-facing rationale `recipe-generate` reads to write logic and
  prose: `problem` (2-3 sentences on what breaks without the pattern), `source_excerpt` (a
  10-25 line excerpt or pseudocode showing it clearest), `structure_outline` (the workflow /
  key activity / tool shape, following `structure.md`), `closest_recipe` (nearest existing
  recipe and what differs), `wishlist_gap` (if any), `size_estimate` (rough line count; flag
  anything over ~400 lines as too complex), and `notes` (an overflow list for anything
  load-bearing that doesn't fit the fields above: a gotcha in the source, a concurrency or
  ordering constraint, a test strategy). Put it in `notes` rather than dropping it.

Keep the controlled-vocabulary fields (`category`, `language`, `provider`) exactly as listed
so the card validates and the rendered front matter is correct by construction.

Write the YAML so it parses: quote any `notes` item, `description`, `closest_recipe`, or
`size_estimate` that contains a colon (a bare `colon: space` makes YAML read the value as a
mapping and the card fails validation), or use a block scalar. See the "Write the YAML so it
parses" section in `proposal-card.md`.

After the cards, add an **Excluded patterns** section: interesting candidates filtered out,
one line each.

To build a recipe from a card, save it under `cards/` as `cards/<name>.card.yaml` (create the
directory if needed; `cards/` is gitignored, so cards are scratch inputs, not committed
artifacts), then pass that path to `/ai-cookbook:recipe-generate`.
