"""NVIDIA NIM serves the OpenAI chat-completions API, so the OpenAI Agents SDK talks to Nemotron with no integration code:
a provider pointed at a different base URL. Retries stay off on the client; the model call is a Temporal Activity and its
retry policy owns retries."""

from __future__ import annotations

import os

from agents import ModelSettings, OpenAIProvider, set_tracing_disabled
from openai import AsyncOpenAI

# The blog's recorded run used nvidia/nemotron-3-super-120b-a12b. Nemotron 3.5 Lightning answers in about a second and
# handles this recipe's tool calls, so it is the default; set NIM_MODEL to use another model served by NIM.
DEFAULT_MODEL = os.environ.get("NIM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")


def nim_model_provider() -> OpenAIProvider:
    set_tracing_disabled(True)  # the SDK's default trace exporter would post to api.openai.com
    return OpenAIProvider(
        openai_client=AsyncOpenAI(
            base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=os.environ.get("NVIDIA_API_KEY", "placeholder"),
            max_retries=0,
        ),
        use_responses=False,  # NIM serves chat completions, not the Responses API
    )


def nemotron_settings() -> ModelSettings:
    """Thinking off: Nemotron then answers a tool call in a few hundred tokens instead of reasoning past the token budget."""
    return ModelSettings(temperature=0.2, max_tokens=2048, parallel_tool_calls=False, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
