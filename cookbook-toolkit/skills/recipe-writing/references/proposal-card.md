# Recipe proposal card

A **proposal card** is the hand-off between `recipe-scout` and `recipe-generate`. It is a
single machine-ingestible YAML file with two blocks:

- **`recipe:`** — deterministic fields. `recipe-scaffold` (the Python/Jinja tool) consumes
  these to render the recipe skeleton — no LLM involved, identical every time, lint-clean.
- **`context:`** — prose/logic inputs. `recipe-generate`'s LLM reads these to write the
  Activity logic and the README walkthrough prose into the scaffolded stubs.

The schema is authoritative: [`card-schema.json`](card-schema.json). `recipe-scaffold`
validates a card against it and fails fast on a malformed card.

## Example

```yaml
recipe:
  name: guardrails-hard-rules        # kebab-case → dir guardrails-hard-rules_python,
                                     #   package cookbook-guardrails-hard-rules-python,
                                     #   task queue guardrails-hard-rules-task-queue
  category: agents                   # agents | foundations | deep_research | mcp
  language: python
  provider: [anthropic]              # 0+ of: openai, anthropic, litellm (first drives the client)
  title: "Guardrails: Hard Rules"    # README H1
  description: A post-LLM guardrail layer where deterministic rules override the LLM verdict.
  priority: 500                      # band per frontmatter.md
  components:                        # optional; defaults used if omitted
    workflow_class: ClassifyContentWorkflow
    activities: [classify]

context:
  problem: >
    LLMs can be prompt-injected or hallucinate, so a model alone can't be trusted for
    decisions with real consequences. Without a deterministic backstop, a manipulated
    verdict ships.
  source_excerpt: |
    def apply_hard_rules(signals, llm_verdict):
        if llm_verdict.classification == "block":
            return llm_verdict
        ...
  structure_outline: >
    Workflow runs one classify Activity; the Activity calls the LLM with a forced tool,
    then applies deterministic hard rules that can override the verdict to "block".
  closest_recipe: tool_call_openai_python — adds a deterministic post-LLM override layer.
  wishlist_gap: Guardrails
  size_estimate: ~250 lines
```

## `recipe:` fields (deterministic)

| Field | Required | Purpose |
| :--- | :--- | :--- |
| `name` | yes | kebab-case; drives directory, package name, task queue. |
| `category` | yes | One of `agents`, `foundations`, `deep_research`, `mcp` (see frontmatter.md). |
| `language` | yes | `python` (the scaffolder's only target today). |
| `provider` | no | Array, 0+ of `openai`/`anthropic`/`litellm`; the first drives the scaffolded client + SDK dependency. |
| `title` | yes | README H1. |
| `description` | yes | One plain sentence; README front-matter `description`. |
| `priority` | yes | Integer; front-matter `priority` (band per frontmatter.md). |
| `components` | no | `workflow_class`, `activities[]`, `tools[]` to stub; sensible defaults otherwise. |

The `category`/`language`/`provider` values are the same controlled vocabulary as
[`tags.json`](tags.json), so the rendered front matter is valid by construction.

## `context:` fields (LLM input)

`problem`, `source_excerpt`, `structure_outline`, `closest_recipe`, `wishlist_gap`,
`size_estimate` — the reviewer-facing rationale `recipe-scout` produces. `recipe-generate`
reads them to fill the scaffolded stubs into a runnable recipe; they are not used by the
deterministic scaffolder.

## Pipeline

```
recipe-scout  → emits a card (this format)
recipe-scaffold (card.recipe → Jinja skeleton, deterministic, lint-clean)
recipe-generate (card.context + stubs → LLM writes Activity logic + README prose → recipe-lint)
```

A contributor without `recipe-scout` can author a card by hand against this format and feed
it to `recipe-generate`.
