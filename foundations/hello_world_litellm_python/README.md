<!-- 
description: Integrate LiteLLM into a durable Temporal Workflow in Python to call and switch between LLM providers.
tags: [foundations, python, litellm]
priority: 920
-->

# Hello world with LiteLLM

[LiteLLM](https://github.com/BerriAI/litellm) is a library for calling LLMs from Python. It makes it easy to access, and switch between, many providers, including OpenAI, Anthropic, Google, and more.

This recipe mirrors the [Basic Python recipe](../hello_world_openai_responses_python/README.md), but swaps the OpenAI SDK for LiteLLM. The Workflow still delegates LLM calls to an Activity, letting Temporal coordinate retries and durability, while LiteLLM forwards those calls to your configured provider.

Key points:

- A reusable Activity that wraps `litellm.acompletion` and keeps retries in Temporal.
- The most common LiteLLM parameters are on `LiteLLMRequest` ensuring type checking and IDE completion. Others may be passed via the `extra_options` dictionary, which functions as `kwargs` for `litellm.acompletion`.
- The Activity returns the full LiteLLM response for processing by the Workflow.

## Create the Activity

`activities/models.py`

<!--SNIPSTART:file activities/models.py-->
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union


@dataclass
class LiteLLMRequest:
    """Lightweight container for the handful of LiteLLM options this sample surfaces."""

    model: str
    messages: List[Dict[str, Any]]

    # Optional knobs: limited to the most common tweaks.
    temperature: Optional[float] = (
        None  # Controls response creativity without extra ceremony.
    )
    max_tokens: Optional[int] = None  # Caps response length/cost.
    timeout: Optional[Union[float, int]] = (
        None  # Lets callers bound slow provider responses.
    )
    response_format: Optional[Union[dict, Type[Any]]] = (
        None  # Hook for JSON/object style responses.
    )

    # Escape hatch for advanced parameters we do not model explicitly.
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def to_acompletion_kwargs(self) -> Dict[str, Any]:
        """Convert this request to kwargs suitable for litellm.acompletion()."""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
        }

        optional_values = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "response_format": self.response_format,
        }

        for key, value in optional_values.items():
            if value is not None:
                kwargs[key] = value

        if self.extra_options:
            kwargs.update(self.extra_options)

        return kwargs
```
<!--SNIPEND-->

`activities/litellm_completion.py`

<!--SNIPSTART:file activities/litellm_completion.py-->
```python
from typing import Any, Dict

import litellm
from temporalio import activity
from temporalio.exceptions import ApplicationError

from activities.models import LiteLLMRequest


@activity.defn(name="activities.litellm_completion.create")
async def create(request: LiteLLMRequest) -> Dict[str, Any]:
    # Temporal best practice: disable LiteLLM retries and let Temporal handle them.
    kwargs = request.to_acompletion_kwargs()
    kwargs["num_retries"] = 0

    try:
        response = await litellm.acompletion(**kwargs)
    except (
        litellm.AuthenticationError,
        litellm.BadRequestError,
        litellm.InvalidRequestError,
        litellm.UnsupportedParamsError,
        litellm.JSONSchemaValidationError,
        litellm.ContentPolicyViolationError,
        litellm.NotFoundError,
    ) as exc:
        raise ApplicationError(
            str(exc),
            type=exc.__class__.__name__,
            non_retryable=True,
        ) from exc
    except litellm.APIError:
        raise

    return response
```
<!--SNIPEND-->

LiteLLM supports many providers. Configure credentials via environment variables (for example `OPENAI_API_KEY`) before running the Activity. For Google-hosted models (Vertex AI or Gemini), the sample relies on the `google-cloud-aiplatform` and `google-auth` dependencies included in `pyproject.toml`; set the usual Google application credentials (`GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, `VERTEXAI_LOCATION`, etc.) so LiteLLM can obtain an access token.

## Create the Workflow

`workflows/hello_world_workflow.py`

<!--SNIPSTART:file workflows/hello_world_workflow.py-->
```python
from datetime import timedelta

from temporalio import workflow

from activities.models import LiteLLMRequest


@workflow.defn
class HelloWorld:
    @workflow.run
    async def run(self, input: str) -> str:
        messages = [
            {"role": "system", "content": "You only respond in haikus."},
            {"role": "user", "content": input},
        ]
        response = await workflow.execute_activity(
            "activities.litellm_completion.create",
            LiteLLMRequest(
                # LiteLLM allows you to switch between models easily
                # model="gpt-4o-mini",
                model="gemini-2.5-flash-lite",
                messages=messages,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )

        message = response["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = "".join(
                part.get("text", "") for part in message if isinstance(part, dict)
            )

        return message
```
<!--SNIPEND-->

Temporal manages Activity retries, so LiteLLM's retry helper is disabled via `num_retries=0`. Use the `extra_options` escape hatch on `LiteLLMRequest` if you need to surface additional LiteLLM parameters without editing the sample.

## Create the Worker

`worker.py`

<!--SNIPSTART:file worker.py-->
```python
import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities import litellm_completion
from workflows.hello_world_workflow import HelloWorld


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="hello-world-python-task-queue",
        workflows=[
            HelloWorld,
        ],
        activities=[
            litellm_completion.create,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```
<!--SNIPEND-->

## Create the Workflow Starter

`start_workflow.py`

<!--SNIPSTART:file start_workflow.py-->
```python
import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.hello_world_workflow import HelloWorld


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    # Submit the Hello World workflow for execution
    result = await client.execute_workflow(
        HelloWorld.run,
        "Tell me about recursion in programming.",
        id="my-workflow-id",
        task_queue="hello-world-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
```
<!--SNIPEND-->

## Running

Start the Temporal Dev Server:

```bash
temporal server start-dev
```

Install dependencies

```bash
uv sync
```

Set the appropriate environment variables before launching the worker (for example `export OPENAI_API_KEY=...` or export `GEMINI_API_KEY=...`) so LiteLLM can reach your chosen provider.

Run the worker:

```bash
uv run worker.py
```

Start the workflow:

```bash
uv run start_workflow.py
```
