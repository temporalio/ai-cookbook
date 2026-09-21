# Recipe Scout

Analyze an external project and identify which parts would make good AI Cookbook recipes.

Usage: `/project:recipe-scout <github-url>`

## What you do

You are an expert at spotting teachable, self-contained AI patterns in real-world projects. Your job is to produce proposal cards that a reviewer — who may never have seen the source project — can use to decide what's worth building into a recipe.

**Audience reminder:** The AI Cookbook targets AI Engineers who are comfortable with LLMs and agents but are new to Temporal. Recipes should teach *AI building blocks* — patterns for how agents think, decide, call tools, and coordinate — with Temporal providing durability underneath. Do NOT propose patterns that are primarily about Temporal orchestration, distributed systems, or infrastructure; those belong in Temporal's own documentation, not here.

---

### Step 1 — Fetch and analyze the project

Fetch the repository at `$ARGUMENTS`. Collect:
- The README (for intent and architecture overview)
- The full file tree (via GitHub API: `https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1`)
- Key source files: LLM integration code, agent/tool patterns, prompt construction, workflow definitions

Look specifically for these **AI building block** patterns, which make strong recipes:
- **Agentic loop** — LLM called in a loop until a stop condition (tool use, stop sequence, empty tool calls)
- **Forced completion** — On the final loop iteration, `tool_choice` is constrained to a specific tool so the agent must commit to a decision rather than looping forever
- **Tool calling** — LLM invokes structured tools; results fed back into the conversation
- **Parallel tool calls** — LLM requests multiple tools simultaneously; all results must be collected before the next turn
- **Multi-agent coordination / agent supervisor** — One agent spawns or delegates to sub-agents; results are aggregated
- **Structured output** — LLM output is parsed and validated against a Pydantic schema
- **Human-in-the-loop** — Workflow pauses and waits for a human decision before continuing
- **Streaming output** — Activity emits incremental tokens/chunks rather than waiting for full completion
- **RAG (retrieval-augmented generation)** — Retrieved context injected into the prompt before calling the LLM
- **Short-term memory** — Conversation history carried across turns within a single workflow run
- **Long-term memory** — Facts or summaries persisted across workflow runs and retrieved on demand
- **Context summarization** — Long conversation history compressed (e.g., via `continue_as_new`) to stay within context limits
- **Guardrails** — LLM output checked against a policy before being acted on; rejected outputs are blocked or re-requested
- **Chain-of-thought / tree-of-thought** — LLM explicitly reasons through steps before producing a final answer
- **Prompt injection prevention** — Untrusted external data is isolated from control instructions (e.g., XML tags, separate message turns)
- **Dynamic system prompts** — System instructions constructed at runtime from context (user prefs, retrieved docs, current state)
- **Cost/token tracking** — Token usage recorded per workflow run for budgeting or rate-limiting
- **Multi-provider LLM abstraction** — Single interface that dispatches to Anthropic, OpenAI, LiteLLM, or local models

Ignore patterns that are primarily about Temporal internals (workflow ID policies, heartbeats, signal/query handlers, replay determinism) unless they are a natural, invisible part of an AI pattern above.

---

### Step 2 — Produce proposal cards

The cookbook has a wishlist of use cases not yet covered. Patterns that fill one of these gaps should be ranked higher:
- RAG pipeline
- Streaming output
- Short-term or long-term memory
- Context summarization (ContinueAsNew)
- Agent supervisor / multi-agent swarm
- Guardrails
- Chain-of-thought / tree-of-thought
- Cost/token tracking
- Trigger-based AI (event-driven or timer-based)
- Web crawler

For each candidate pattern you find, evaluate:
1. **Is it an AI building block?** Would an AI engineer recognize this as a useful pattern for their LLM/agent work, independent of what orchestrator they use?
2. **Is it well-engineered, not a demo?** The cookbook publishes reference-quality code, not flashy one-offs.
3. **Is it self-contained?** Can it stand alone as a 200–400 line recipe without pulling in the entire project?
4. **Is it teachable?** Does it demonstrate a single clear concept a developer can learn from?
5. **Is it novel vs. existing recipes?** Check existing recipes in this repo (foundations/, agents/, deep_research/, mcp/).
6. **Does it fill a wishlist gap?** Cross-reference against the coverage wishlist above.

Rank the top 2–4 patterns. For each, write a proposal card with the following sections — written so a reviewer who has never seen the source project can evaluate it:

**Proposed recipe:** `{category}/{recipe-name}_python`

**One-line description:** _(the README front matter `description` field)_

**The problem it solves:** In 2–3 sentences: what goes wrong if a developer doesn't know this pattern? What mistake do they typically make, and what does that cost them?

**The pattern in the source:** A short code excerpt (10–25 lines) from the source project that shows the pattern at its clearest. If the source isn't Python or doesn't translate directly, show equivalent pseudocode. This is the "exhibit A" that justifies the recipe.

**How the recipe would be structured:** A brief outline — what the workflow does, what the key activity does, what tool or API is involved. Not full code, 5–10 bullet points.

**Closest existing recipe and what's different:** Name the most similar recipe already in the cookbook and state specifically what this adds or changes. If there's no close match, say so.

**Wishlist gap filled:** Which item from the coverage wishlist does this address, if any? If none, say so.

**Estimated size:** Rough line count for the finished recipe (all files combined). Flag anything over 400 lines as potentially too complex for a single recipe.

---

After the proposal cards, add an **Excluded patterns** section listing any patterns that were interesting but filtered out, with a one-line reason for each.

To generate a recipe from one of these proposals, use `/project:recipe-ify` and paste in the proposal card.
