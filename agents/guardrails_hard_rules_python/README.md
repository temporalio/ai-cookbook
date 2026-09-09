<!--
description: Build a durable content-moderation guardrail in Python with Temporal and Claude that layers deterministic hard rules over an LLM's verdict for auditable overrides.
tags: [agents, python, anthropic]
priority: 500
-->

# Post-LLM guardrail with hard-rule overrides

This recipe shows how to combine an LLM classifier with a deterministic guardrail layer. The LLM provides nuanced judgment for ambiguous cases; hard rules act as a safety net for unambiguous policy violations, overriding the LLM's verdict regardless of what it concluded.

The pattern answers a real problem: LLMs can be manipulated via prompt injection or hallucinate outright. For any decision with real consequences — content moderation, access control, transaction approval — you shouldn't rely on the LLM alone. Hard rules catch clear-cut cases deterministically; the LLM handles everything in the grey zone. Critically, when a hard rule fires, the LLM's original reasoning is preserved inside the override so every decision remains auditable.

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

## Architecture

- **Models** (`models/`):
  - `signals.py`: `ContentSignals` — the text and metadata being classified
  - `verdict.py`: `LLMVerdict` (the LLM's raw classification, also used as the tool's input schema) and `Verdict` (adds `overridden_by_hard_rule`)
- **Guardrails** (`guardrails/hard_rules.py`): pure functions that check content against banned keywords, phone numbers, and email addresses, and escalate the verdict to `block` when one matches
- **Activity** (`activities/classify.py`): calls Claude via a forced tool call to get a structured `LLMVerdict`, then applies the hard rules
- **Workflow** (`workflows/classify_workflow.py`): orchestrates the single `classify` Activity call with a 3-attempt retry policy
- **Scripts**:
  - `worker.py`: runs the Temporal Worker
  - `start_workflow.py`: runs the two examples shown in [Expected output](#expected-output)

## Key patterns

### Forcing a structured verdict from the LLM

The Activity forces Claude to call a single tool, so the response is always a well-formed `LLMVerdict` instead of free-form text to parse:

<!--SNIPSTART activities/classify.py {"startPattern": "^_SUBMIT_VERDICT_TOOL = \\{$", "endPattern": "^\\}$"}-->
```python
_SUBMIT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit your content moderation classification.",
    "input_schema": LLMVerdict.model_json_schema(),
}
```
<!--SNIPEND-->

<!--SNIPSTART activities/classify.py {"startPattern": "response = await client\\.messages\\.create\\($", "endPattern": "^\\s*\\)\\s*$", "selectedLines": ["1", "8-10"]}-->
```python
response = await client.messages.create(
    ...
    tools=[_SUBMIT_VERDICT_TOOL],
    tool_choice={"type": "tool", "name": "submit_verdict"},
)
```
<!--SNIPEND-->

### Overriding while preserving the original reasoning

`apply_hard_rules` never discards the LLM's own verdict — a rule can only escalate a verdict to `block`, and when it does, the LLM's reasoning is embedded in the result so the override stays auditable:

<!--SNIPSTART guardrails/hard_rules.py {"startPattern": "^def apply_hard_rules\\(signals: ContentSignals, llm_verdict: Verdict\\) -> Verdict:$", "endPattern": "^\\s*\\)\\s*$"}-->
```python
def apply_hard_rules(signals: ContentSignals, llm_verdict: Verdict) -> Verdict:
    """Post-filter: override the LLM verdict if a hard rule matches.

    When a rule fires, the LLM's original reasoning is embedded in the
    returned verdict so the override is auditable.
    """
    if llm_verdict.classification == "block":
        return llm_verdict

    hard = _hard_block(signals)
    if hard is None:
        return llm_verdict

    return Verdict(
        classification=hard.classification,
        confidence=hard.confidence,
        overridden_by_hard_rule=True,
        reasoning=(
            f"{hard.reasoning}\n\n"
            f"[LLM classified as '{llm_verdict.classification}' — "
            f"reasoning: {llm_verdict.reasoning}]"
        ),
    )
```
<!--SNIPEND-->

### Non-retryable Anthropic errors

Permanent client errors (bad request, authentication, permission-denied, not-found, unprocessable-entity) are classified as non-retryable so the Workflow doesn't keep retrying a request that can never succeed:

<!--SNIPSTART activities/classify.py {"startPattern": "^\\s*except \\($", "endPattern": "\\) from exc$"}-->
```python
except (
    anthropic.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.UnprocessableEntityError,
) as exc:
    # These errors will never succeed on a retry: a retired model identifier, for
    # example, returns 404. Retrying them under the default policy would loop
    # forever instead of surfacing the problem. Everything else, such as rate
    # limits and 5xx responses, propagates so Temporal can retry it.
    raise ApplicationError(
        str(exc),
        type=exc.__class__.__name__,
        non_retryable=True,
    ) from exc
```
<!--SNIPEND-->

## Extensions

This pattern can be extended to:
- Add more hard rules (regex, allow/deny lists, PII detectors) without touching the LLM prompt
- Log every override to a compliance or audit sink for review
- Route `review` verdicts to a human approval step — see [Human-in-the-loop AI agent](../human_in_the_loop_python)
