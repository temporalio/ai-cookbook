# What is in scope for a cookbook recipe

This is the single source of truth for judging whether a pattern belongs in the cookbook at
all. `structure.md`, `layout.md`, and `code-conventions.md` answer "is this recipe built
correctly"; this file answers the prior question: "is this an AI Cookbook recipe in the first
place."

## Audience and purpose

The cookbook targets AI engineers who are comfortable with LLMs and agents but new to
Temporal. A recipe teaches an **AI building block** (how an agent thinks, decides, calls
tools, coordinates, remembers, or recovers) and shows it built durably on Temporal. The AI
pattern is the hero; Temporal is the durability layer underneath.

So the whole point of the cookbook is to show known agent patterns *implemented in Temporal*.
A recipe that does that well is exactly what we want.

## The test: what is the recipe ABOUT, not which primitives it uses

The one rule people get wrong: the exclusion is about a recipe's **subject**, not its
**ingredients**.

- **In scope:** the headline concept is an AI building block. The recipe is free to use any
  Temporal primitive it needs: continue-as-new, child workflows, signals, timers, queries,
  activities, retry policies. Which primitives appear says nothing about scope.
- **Out of scope:** the headline concept *is* a Temporal mechanic, with no agent concept on
  top of it. These belong in Temporal's own docs.

Do not reason "this uses signals (or child workflows, or continue-as-new), so it's a Temporal
pattern." That inference is wrong and has caused good recipes to be rejected. Ask what the
reader walks away having learned.

### Two questions to apply it

1. **Headline test.** Write the recipe's lesson as one sentence. Is the *subject* of that
   sentence how an agent thinks / decides / calls tools / coordinates / remembers / recovers?
   Or is the subject a Temporal primitive? Subject = agent concept → in scope. Subject =
   Temporal mechanic → out.
2. **Orchestrator-swap test.** If you swapped Temporal for another durable-execution engine,
   would the *lesson* survive (in scope) or vanish (out)? This asks whether the concept is
   orchestrator-independent. It does **not** ask whether the implementation avoids Temporal
   features. A steerable agent still teaches a portable concept even though it rides on
   Temporal signals.

## Worked examples

These are in scope even though each leans on a heavyweight Temporal primitive, because an
agent concept is the headline and the primitive is invisible plumbing:

| Recipe concept | Temporal primitive it rides on | Why it's in scope |
| :--- | :--- | :--- |
| Context summarization / bounded agent memory | continue-as-new | Lesson is memory compaction; CAN is the carry-forward mechanism. |
| Multi-agent supervisor / subagent delegation | child workflows | Lesson is task delegation across agents; child workflows are the durable boundary. |
| Steerable agent (nudge mid-run) / human-in-the-loop | signals, cancellation | Lesson is injecting guidance or approval into a live loop; signals are the transport. |
| Parallel tool calls | concurrent activities | Lesson is fanning out tool calls in one turn; activities are how each runs durably. |

These are out of scope, because the recipe would be *about* the Temporal mechanic with no AI
building block on top:

- Workflow-ID reuse / deduplication policy.
- Heartbeat tuning and liveness for long activities.
- Replay determinism debugging, worker versioning, task-queue routing.
- Activity retry-policy or idempotency-key design treated as the topic itself (fold
  idempotency into a mutating-tool recipe as a note instead).

A pattern can be genuinely interesting and still be out of scope here; when it is, note it in
the Excluded section with the one-line reason it belongs in Temporal's docs rather than the
cookbook.
