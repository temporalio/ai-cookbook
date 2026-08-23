<!--
description: Build a durable fixed-flow AI agent pipeline in Python with Temporal and the OpenAI Responses API.
tags: [agents, python, openai]
priority: 650
-->

# Fixed-flow AI agents

This recipe runs three AI specialists in a predetermined sequence: an analyst extracts the important evidence, a critic finds gaps and risks, and a decision specialist writes the final recommendation. The Workflow decides which specialist runs next; the model cannot skip, repeat, or add stages.

A fixed flow is useful when the process is known in advance and needs to be repeatable, observable, and easy to test. Use a dynamic agentic loop instead when the model genuinely needs to choose its next tool or specialist at runtime.

Each model call is a Temporal Activity. If a Worker or model request fails, Temporal retries only the failed stage and preserves the completed stages in Event History.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A running Temporal server: `temporal server start-dev`
- `OPENAI_API_KEY` environment variable set

## Run it

```bash
uv sync

# Terminal 1 — start the worker
uv run worker.py

# Terminal 2 — start the fixed flow
uv run start_workflow.py
```

The example evaluates a proposal to extend a neighborhood library's weekend hours and prints the output from every stage:

```text
Analysis
--------
...

Critique
--------
...

Recommendation
--------------
...
```

## Architecture

- **Models** (`models/flow.py`): typed inputs and outputs for the Workflow and each specialist stage
- **Activity** (`activities/run_agent_stage.py`): runs one OpenAI Responses API call with client-side retries disabled
- **Workflow** (`workflows/fixed_flow_workflow.py`): defines and executes the analysis, critique, and recommendation stages in order
- **Worker** (`worker.py`): registers the Workflow and Activity
- **Starter** (`start_workflow.py`): submits an example and prints the results
- **Tests** (`tests/test_fixed_flow.py`): verify client cleanup, retryable empty responses, stage ordering, and context propagation without calling OpenAI

## Key patterns

### Keep routing in the Workflow

The sequence is visible in Workflow code and Event History. Each stage receives the earlier output it needs, but no stage decides what runs next:

<!--SNIPSTART workflows/fixed_flow_workflow.py:fixed-sequence-->
```python
analysis = await workflow.execute_activity(
    run_agent_stage,
    AgentStageRequest(
        stage="analysis",
        model=request.model,
        instructions=(
            "You are an analysis specialist. Extract the important claims, "
            "evidence, and assumptions from the source material. Do not make "
            "a final recommendation."
        ),
        input=f"Topic: {request.topic}\n\nSource material:\n{request.source_material}",
    ),
    start_to_close_timeout=timedelta(seconds=45),
    retry_policy=ACTIVITY_RETRY_POLICY,
)

critique = await workflow.execute_activity(
    run_agent_stage,
    AgentStageRequest(
        stage="critique",
        model=request.model,
        instructions=(
            "You are a critical reviewer. Identify unsupported claims, missing "
            "information, contradictions, and risks in the analysis. Do not "
            "rewrite it or make the final recommendation."
        ),
        input=(
            f"Topic: {request.topic}\n\n"
            f"Source material:\n{request.source_material}\n\n"
            f"Analysis to review:\n{analysis.output}"
        ),
    ),
    start_to_close_timeout=timedelta(seconds=45),
    retry_policy=ACTIVITY_RETRY_POLICY,
)

recommendation = await workflow.execute_activity(
    run_agent_stage,
    AgentStageRequest(
        stage="recommendation",
        model=request.model,
        instructions=(
            "You are a decision specialist. Produce a concise recommendation "
            "with rationale, uncertainties, and concrete next steps. Base it "
            "only on the supplied source, analysis, and critique."
        ),
        input=(
            f"Topic: {request.topic}\n\n"
            f"Source material:\n{request.source_material}\n\n"
            f"Analysis:\n{analysis.output}\n\n"
            f"Critique:\n{critique.output}"
        ),
    ),
    start_to_close_timeout=timedelta(seconds=45),
    retry_policy=ACTIVITY_RETRY_POLICY,
)

return FixedFlowResult(
    analysis=analysis.output,
    critique=critique.output,
    recommendation=recommendation.output,
)
```
<!--SNIPEND-->

This makes the orchestration replay-safe and gives every model call an independent timeout and retry boundary. To change the process, edit the Workflow explicitly instead of changing a routing prompt.

### Let Temporal own retries

The OpenAI client has retries disabled. Permanent client errors are marked non-retryable, while transient failures and empty responses are left retryable under the Workflow's three-attempt Activity policy:

<!--SNIPSTART activities/run_agent_stage.py {"startPattern": "^@activity\\.defn$", "endPattern": "^    return AgentStageResult\\(stage=request\\.stage, output=output\\)$"}-->
```python
@activity.defn
async def run_agent_stage(request: AgentStageRequest) -> AgentStageResult:
    """Run one specialist role as a retryable Temporal Activity."""

    client = AsyncOpenAI(max_retries=0)
    try:
        response = await client.responses.create(
            model=request.model,
            instructions=request.instructions,
            input=request.input,
            timeout=30,
        )
    except (
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.UnprocessableEntityError,
    ) as exc:
        raise ApplicationError(
            str(exc),
            type=exc.__class__.__name__,
            non_retryable=True,
        ) from exc
    finally:
        await client.close()

    output = response.output_text.strip()
    if not output:
        # An empty response may be transient, so let the Workflow's Activity retry
        # policy decide whether to try the stage again.
        raise ApplicationError(
            f"The {request.stage} stage returned no text.",
            type="EmptyModelResponse",
        )

    return AgentStageResult(stage=request.stage, output=output)
```
<!--SNIPEND-->

## When to use this pattern

Choose a fixed flow when:

- every case should pass through the same review stages
- reviewers need to see which stage produced each output
- a failed stage should retry without re-running successful earlier stages
- tests must assert the exact route through the process

Choose model-directed routing when the useful set or order of steps cannot be known until runtime.

This recipe extracts the fixed-flow pattern from Temporal's larger [document-processing example](https://github.com/temporal-sa/document-processing-examples), which demonstrates OCR, specialist analysis, deterministic policy checks, and human review in a mortgage-underwriting scenario.
