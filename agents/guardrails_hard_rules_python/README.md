<!--
description: Demonstrates a post-LLM guardrail layer that uses deterministic hard rules to override an LLM's verdict, ensuring policy-critical decisions can never be bypassed by hallucination or prompt injection.
tags: [agents, python, anthropic]
priority: 500
-->

# Guardrails: Hard Rules

This recipe shows how to combine an LLM classifier with a deterministic guardrail layer. The LLM provides nuanced judgment for ambiguous cases; hard rules act as a safety net for unambiguous policy violations, overriding the LLM's verdict regardless of what it concluded.

The pattern answers a real problem: LLMs can be manipulated via prompt injection or simply hallucinate. For any decision with real consequences — content moderation, access control, transaction approval — you shouldn't rely on the LLM alone. Hard rules catch clear-cut cases deterministically; the LLM handles everything in the grey zone. Critically, when a hard rule fires, the LLM's original reasoning is preserved inside the override so every decision remains auditable.

The recipe uses a content moderation scenario: user-submitted text is classified as `safe`, `review`, or `block`. Hard rules override to `block` when contact information or banned keywords are detected, regardless of what the LLM concluded.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A running Temporal server: `temporal server start-dev`
- `ANTHROPIC_API_KEY` environment variable set

## Run it

```bash
uv sync

# Terminal 1 — start the worker
uv run python -m worker

# Terminal 2 — submit two example workflows
uv run python -m start_workflow
```

## Expected output

```
--- Example 1: Hard rule override ---
Input: 'Great product! Contact me at john.doe@example.com for a special deal.'
Classification: block
Overridden by hard rule: True
Reasoning: Hard rule: contains email address (privacy policy violation).

[LLM classified as 'safe' — reasoning: The message is promotional but does not appear harmful.]

--- Example 2: LLM verdict stands ---
Input: 'I really enjoyed the hiking trail last weekend. The views were amazing!'
Classification: safe
Overridden by hard rule: False
Reasoning: Positive personal experience with no policy concerns.
```

In Example 1, the LLM's classification and reasoning are preserved inside brackets — the override is fully auditable.
