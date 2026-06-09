<!--
description: RECIPE_DESCRIPTION
tags: [RECIPE_CATEGORY, python, RECIPE_PROVIDER]
priority: RECIPE_PRIORITY
-->

# RECIPE_TITLE

One or two sentences on what this recipe demonstrates and the pattern it teaches. The LLM
call happens in a Temporal Activity, so Temporal handles retries and durability.

## Create the Activity

Describe what the Activity does and why. We set `max_retries=0` so Temporal owns retries,
not the client, and re-raise permanent API errors as a non-retryable `ApplicationError`.

*File: activities/llm_call.py*

```python
@activity.defn
async def call_llm(request: CallLLMRequest) -> str:
    client = AsyncOpenAI(max_retries=0)
    response = await client.responses.create(model=request.model, input=request.prompt)
    return response.output_text
```

## Create the Workflow

The Workflow is pure orchestration: it calls the Activity with an explicit
`start_to_close_timeout` and returns the result.

*File: workflows/recipe_workflow.py*

```python
@workflow.defn
class RecipeWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return await workflow.execute_activity(
            call_llm,
            CallLLMRequest(prompt=prompt),
            start_to_close_timeout=timedelta(seconds=30),
        )
```

## Create the Worker

The Worker registers the Workflow and Activity on the task queue. It uses
`pydantic_data_converter` so Pydantic/dataclass payloads serialize correctly.

*File: worker.py*

## Create the Workflow Starter

The starter submits one Workflow execution and prints the result.

*File: start_workflow.py*

## Running

```bash
temporal server start-dev            # if not already running
uv sync
uv run python -m worker              # terminal 1
uv run python -m start_workflow      # terminal 2
```
