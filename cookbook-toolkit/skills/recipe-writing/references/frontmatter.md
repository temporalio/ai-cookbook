# Recipe front matter

Front matter lives **only on a recipe's `README.md`**: never in code files. It is a
single HTML comment at the very top of the file. The docs sync (`bin/sync-ai-cookbook.js`)
parses it and merges `description`, `tags`, and `priority` into the published page.

## Schema

```markdown
<!--
description: One plain-text sentence describing what the recipe demonstrates.
tags: [category, language, provider]
priority: 500
-->
```

| Field | Required | Rule |
| :--- | :--- | :--- |
| `description` | Yes (hard) | One plain-text sentence. Missing → breaks the docs page; a hard error. |
| `tags` | Yes (warn) | Controlled vocabulary, ordered. See below. |
| `priority` | Yes (warn) | Integer. Orders the cookbook category index (higher = earlier). |

`description` is the only field whose absence is a **hard error** (it, plus a missing H1
and invalid YAML, are the only docs-breakers). Everything else is a **warning**: reported,
not blocking.

## Why an HTML comment, not `---` YAML front matter

Two fixed constraints, not preferences:

1. **GitHub rendering**: a recipe README is browsed directly on GitHub, where a `---`
   front-matter block renders as an ugly horizontal rule / table at the top. An HTML
   comment is invisible there.
2. **The docs sync parses the comment**: `bin/sync-ai-cookbook.js` extracts the
   `<!-- … -->` block specifically, and changing the sync mechanism is a non-goal.

So the HTML-comment form stays.

## Must be valid YAML

The body of the comment must parse as YAML under `js-yaml` (what the docs build uses).
Invalid YAML is a hard error. The cookbook validator parses with `js-yaml` too, so CI and
the docs build agree.

## Tags

The accepted vocabulary is defined in [`tags.json`](tags.json): that file is
**authoritative** (the validator and `recipe-lint` import it; this prose just describes
it). A tag is valid if, after synonym resolution, it appears in `categories`, `languages`,
or `providers`.

- **Spacing**: `tags: [agents, python, openai]`: one space after the colon, items
  comma-space separated. `tags:[…]` (no space) is a warning.
- **Order**: `category, language, provider(s)`: in that order.
- **Category** (required, exactly one): `agents`, `foundations`, `deep_research`, `mcp`.
  Must match the recipe's directory category.
- **Language** (required): `python` (future: `typescript`, `go`).
- **Provider** (optional, zero or more): `openai`, `anthropic`, `litellm`. Omit when the
  recipe has no LLM provider (e.g. an MCP weather server).
- **Synonyms** (auto-flagged, fix to canonical): `claude` → `anthropic`,
  `provider-neutral` → `litellm`.
- **Unknown tags are warnings.** The topic one-offs currently in the corpus
  (`toolcalling`, `claim-check`, `s3`, `workflows`) are **not** in the vocabulary, remove
  them. (A broader, Temporal-wide tag vocabulary is a separate effort; see plan Open items.
  Keep the cookbook list small until then.)

## Priority bands

`priority` is an integer that orders the cookbook's category index (higher appears
earlier). Keep the **existing values**: these bands describe where current recipes already
sit; do not churn them:

| Band | Range | Holds |
| :--- | :--- | :--- |
| Foundations / hello-world | 900-999 | `hello_world_openai_responses` (999), `structured_output` (980), `hello_world_litellm` (920), `http_retry` (920) |
| Agents / tool-calling / MCP | 700-799 | agentic-loop recipes, `tool_call_openai`, `human_in_the_loop`, `openai_agents_sdk` (750-775), `mcp` (775) |
| Advanced / multi-step | 300-599 | `guardrails` (500), `claim_check` (400), `deep_research/basic_openai` (399) |

Bands are **guidance for choosing a value**, not a hard per-category rule, `claim_check`
lives in `foundations/` but sits in the advanced band intentionally. The validator only
requires that `priority` is present and an integer.

## Exclusions

Never put `last_updated` or `title` in front matter. Both are derived by the docs sync
(`title` from the H1; `last_updated` from git author date). Including them is a warning.

## Current violations (Step 18 worklist)

From the corpus inventory:

- **Spacing `tags:[` (no space)**: `claim_check_pattern_python`, `hello_world_litellm_python`,
  `hello_world_openai_responses_python`, `http_retry_enhancement_python`,
  `structured_output_openai_responses_python`.
- **Synonyms to canonicalize**: `agentic_loop_tool_call_claude_python` (`claude` →
  `anthropic`); `hello_world_litellm_python` (`provider-neutral` → `litellm`).
- **Unknown topic tags to remove**: `basic_openai_python` (`toolcalling`),
  `claim_check_pattern_python` (`claim-check`, `s3`), `hello_world_durable_mcp_server`
  (`workflows`).
- **Category tag wrong/missing**: `deep_research/basic_openai_python` is tagged `agents`
  (should be `deep_research`); `hello_world_litellm_python` has no category tag (should
  lead with `foundations`).
- **Ordering**: fix once categories are added (e.g. `hello_world_litellm_python` should be
  `[foundations, python, litellm]`).
