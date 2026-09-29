# Temporal Workflows

A Workflow is a sequence of steps that orchestrate Activities and child Workflows.

## What is a Workflow?

Workflows are the fundamental unit of a Temporal Application. They are durable functions that execute reliably even when failures occur.

## Key Features

- **Durable**: Workflows can run for seconds, days, or even years
- **Reliable**: Automatic recovery from failures
- **Testable**: Full deterministic replay capabilities
- **Observable**: Complete execution history

## Writing a Workflow

In Python, define a workflow using the `@workflow.defn` decorator:

```python
from temporalio import workflow
from datetime import timedelta

@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        result = await workflow.execute_activity(
            my_activity,
            name,
            start_to_close_timeout=timedelta(seconds=30),
        )
        return f"Hello {result}"
```

## Workflow Rules

- Workflows must be deterministic
- Use Activities for non-deterministic operations
- Don't use random numbers or timestamps directly
- Always use workflow.now() for current time

## Signals and Queries

Workflows can receive external input via signals and respond to queries:

- **Signals**: Asynchronous messages that modify workflow state
- **Queries**: Synchronous read-only requests for workflow state
