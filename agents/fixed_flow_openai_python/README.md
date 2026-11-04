<!--
description: Build a bounded, policy-constrained mortgage underwriting agent with Temporal and OpenAI.
tags: [agents, python, openai, human-in-the-loop]
priority: 650
-->

# Policy-constrained fixed-flow agent

Having trouble ensuring an LLM review never skips required checks, never auto-approves
a hard-policy violation, and eventually stops or escalates to a human? This recipe uses
a small mortgage underwriting example where Temporal owns the route:

1. an analyst produces a typed assessment
2. a critic either accepts it or requests a revision
3. the Workflow permits at most two revisions
4. a decision agent produces a typed recommendation
5. deterministic policy checks can force human review

Each model call is an Activity, so Temporal retries only the failed call and preserves
completed work. The review packet is exposed by Query; a reviewer resumes the Workflow
with a Signal.

> The thresholds and application schema are intentionally simplified teaching data,
> not production lending policy. The example sends no personal identifiers to OpenAI.

This is a Cookbook-sized extraction of the fixed-flow mortgage example in
[document-processing-examples](https://github.com/temporal-sa/document-processing-examples).
It deliberately leaves out OCR, PDFs and retrieval, a web UI, datasets, and most of the
domain schema so the reliability controls remain visible.

## Run it

Prerequisites: Python 3.10+, [uv](https://docs.astral.sh/uv/), an
`OPENAI_API_KEY`, and a local server started with `temporal server start-dev`.

```bash
uv sync

# terminal 1
uv run worker.py

# terminal 2: inspect the review packet, then approve or reject it
uv run start_workflow.py
```

The starter uses an example credit score below the demo policy threshold, so the
Workflow always pauses at the human gate.

## Bounded orchestration

The model never selects the next stage. The Workflow records the fixed route and the
revision counter in Event History:

<!--SNIPSTART workflows/fixed_flow_workflow.py:bounded-fixed-flow-->
```python
analysis = await workflow.execute_activity(
    analyze_application,
    AgentTask(application=application, metrics=metrics, model=request.model),
    start_to_close_timeout=ACTIVITY_TIMEOUT,
    retry_policy=RETRY_POLICY,
)

critique = CriticReview(accepted=False, issues=[], revision_instructions=[])
for revision_count in range(MAX_REVISIONS + 1):
    critique = await workflow.execute_activity(
        critique_analysis,
        AgentTask(
            application=application,
            metrics=metrics,
            previous_analysis=analysis,
            model=request.model,
        ),
        start_to_close_timeout=ACTIVITY_TIMEOUT,
        retry_policy=RETRY_POLICY,
    )
    if critique.accepted or revision_count == MAX_REVISIONS:
        break
    analysis = await workflow.execute_activity(
        analyze_application,
        AgentTask(
            application=application,
            metrics=metrics,
            previous_analysis=analysis,
            critique=critique,
            model=request.model,
        ),
        start_to_close_timeout=ACTIVITY_TIMEOUT,
        retry_policy=RETRY_POLICY,
    )

recommendation = await workflow.execute_activity(
    draft_decision,
    AgentTask(
        application=application,
        metrics=metrics,
        previous_analysis=analysis,
        critique=critique,
        model=request.model,
    ),
    start_to_close_timeout=ACTIVITY_TIMEOUT,
    retry_policy=RETRY_POLICY,
)
```
<!--SNIPEND-->

The OpenAI Responses API parses each stage directly into a Pydantic model. OpenAI SDK
retries are disabled; Temporal retries transient Activity failures up to three times,
while permanent request and authentication failures are marked non-retryable.

## Deterministic policy and human review

Two example rules run in normal Python: credit score must be at least 620 and
debt-to-income must not exceed 50%. These checks are independent of the LLM. A violation,
an unresolved critic objection, or an explicit model request all create the same review
gate:

<!--SNIPSTART workflows/fixed_flow_workflow.py:policy-human-gate-->
```python
self._review_packet = HumanReviewPacket(
    application=application,
    metrics=metrics,
    analysis=analysis,
    critique=critique,
    recommendation=recommendation,
    policy_violations=violations,
    review_reasons=reasons,
)
await workflow.wait_condition(lambda: self._human_review is not None)
human_review = self._human_review
final_decision = human_review.decision
```
<!--SNIPEND-->

This composition is the point of the recipe: structured model output alone does not
guarantee process completeness, and durable human input alone does not stop an agent
from wandering. The fixed route, bounded revision loop, deterministic override, Query,
and Signal work together to make the decision traceable and guaranteed to terminate or
wait at an explicit human boundary.
