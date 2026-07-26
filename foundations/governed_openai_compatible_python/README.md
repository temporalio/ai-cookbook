<!--
description: Run an OpenAI-compatible model call as a durable Temporal Activity with stable workflow and request correlation IDs.
tags: [foundations, python, openai, temporal]
priority: 500
-->

# Governed OpenAI-Compatible Activity

This recipe shows a durable Temporal workflow that calls an
OpenAI-compatible endpoint from an Activity. It uses the standard OpenAI
Python SDK, so the same pattern works with a self-hosted model endpoint, an
enterprise gateway, or the OpenAI API.

The example keeps two responsibilities separate:

- **Temporal** owns workflow state, Activity retries, and recovery.
- **The model endpoint** owns model routing and any endpoint-side controls,
  such as access policy, usage accounting, or request tracing.

The Activity disables SDK retries with `max_retries=0`. Temporal's Activity
retry policy remains the single retry authority.

## Correlation

The workflow ID becomes a stable run ID. Each model Activity derives a stable
request ID and sends both as headers. An endpoint that does not recognize the
headers simply ignores them. Gateways that support correlation can use them to
join model usage to the Temporal workflow without changing the workflow's
business logic.

## Configure

Set the endpoint and key before starting the worker:

```bash
export OPENAI_BASE_URL=https://api.example.com/v1
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL=gpt-4o
```

For Tuning Engines, use an inference key and the governed proxy endpoint:

```bash
export OPENAI_BASE_URL=https://api.tuningengines.com/v1
export OPENAI_API_KEY=sk-te-YOUR-INFERENCE-KEY
export OPENAI_MODEL=gpt-4o
```

Tuning Engines records the call with the same workflow-derived run and request
IDs, so it can be correlated with policy decisions, approvals, usage, and
runtime traces. The Temporal workflow continues to own durable execution and
retry behavior.

## Run

Start a local Temporal server:

```bash
temporal server start-dev
```

Install dependencies, then start the worker in one terminal:

```bash
uv sync
uv run python -m worker
```

Start the workflow in another:

```bash
uv run python -m start_workflow
```

## Test

The test suite replaces the external Activity with a local mock, so no model
endpoint or API key is required:

```bash
uv run pytest tests/
```
