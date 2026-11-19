"""NVIDIA NIM serves the OpenAI chat-completions API, so the OpenAI Agents SDK talks to Nemotron with no integration code:
a provider pointed at a different base URL. Every model call goes through NeMo Relay's managed LLM pipeline (relay.py), and
a provider-side failure is routed to a fallback Nemotron model instead of failing the Activity. Which model answered is in
the Relay trajectory (a `model_fallback` mark and the LLM summary) and in the Worker log. Client retries stay off: the
model call is a Temporal Activity and its retry policy owns retries."""

from __future__ import annotations

import inspect
import os
from typing import Any

from agents import ModelSettings, OpenAIProvider, set_tracing_disabled
from agents.models.interface import Model
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from temporalio import activity

import relay

# The recorded demo ran nvidia/nemotron-3-super-120b-a12b with Nemotron 3.5 Lightning as the fallback. Lightning answers
# this recipe's tool calls in about a second, so it is the default; set NIM_MODEL and NIM_FALLBACK_MODEL for other pairs.
DEFAULT_MODEL = os.environ.get("NIM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
DEFAULT_FALLBACK = os.environ.get("NIM_FALLBACK_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
FORWARD_HEADERS = ("traceparent", "tracestate")  # what Relay adds to the request that the NIM call should carry


class InjectedFault(Exception):
    """FAULT_MODEL_CALLS=3,7 makes the 3rd and 7th primary model calls of this process fail: the demo's fallback beat."""


_calls = {"n": 0}


def _fault_due() -> bool:
    spec = os.environ.get("FAULT_MODEL_CALLS", "")
    if not spec:
        return False
    _calls["n"] += 1
    return str(_calls["n"]) in {x.strip() for x in spec.split(",") if x.strip()}


def _worth_fallback(e: Exception) -> bool:
    """Only provider-side trouble is routed to the fallback model; bad requests and our own bugs still fail the Activity.
    404 is included: hosted NIM has answered "function not found" for a listed model while another model served normally."""
    if isinstance(e, (InjectedFault, APIConnectionError)):  # APITimeoutError is an APIConnectionError
        return True
    return isinstance(e, APIStatusError) and e.status_code in (404, 408, 409, 429, 500, 502, 503, 504)


_PARAMS = [p for p in inspect.signature(Model.get_response).parameters if p != "self"]


def _bind(args: tuple, kwargs: dict) -> dict[str, Any]:
    """get_response arguments by name, however the caller passed them."""
    kw = dict(zip(_PARAMS, args))
    kw.update(kwargs)
    return kw


def _with_headers(settings: ModelSettings | None, headers: dict[str, str]) -> ModelSettings | None:
    if not headers:
        return settings
    settings = settings or ModelSettings()
    return settings.resolve(ModelSettings(extra_headers={**(settings.extra_headers or {}), **headers}))


def _request_summary(kw: dict[str, Any]) -> dict[str, Any]:
    inp = kw.get("input")
    out: dict[str, Any] = {"tools": [getattr(t, "name", str(t))[:40] for t in (kw.get("tools") or [])][:12]}
    if isinstance(inp, str):
        out["input"] = {"text": inp[:200]}
    elif isinstance(inp, list):
        last = inp[-1] if inp else None
        out["input"] = {"items": len(inp), "last": (last.get("type") or last.get("role")) if isinstance(last, dict) else type(last).__name__}
        calls = [i.get("name") for i in inp if isinstance(i, dict) and i.get("type") == "function_call" and i.get("name")]
        if calls and isinstance(last, dict) and last.get("type") == "function_call_output":
            out["input"]["last_tool"] = calls[-1]  # the tool result this turn follows
    return out


def _response_summary(resp: Any) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    items = getattr(resp, "output", None) or []
    calls = [getattr(i, "name", None) for i in items if getattr(i, "type", "") == "function_call"]
    return {"usage": {k: getattr(usage, k, None) for k in ("requests", "input_tokens", "output_tokens", "total_tokens")} if usage else None,
            "output_items": len(items), "tool_calls": [c for c in calls if c]}


# @@@SNIPSTART relay-model
class RelayModel(Model):
    """Model proxy: the call goes through NeMo Relay's managed LLM pipeline, which records the request summary, the
    response summary and Relay's trace headers; a provider-side failure is answered by the fallback model, with a
    `model_fallback` mark in the trajectory."""

    def __init__(self, inner: Model, name: str, fallback: Model | None, fallback_name: str | None) -> None:
        self._inner, self._name, self._fallback, self._fallback_name = inner, name, fallback, fallback_name

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        kw = _bind(args, kwargs)
        chosen: dict[str, Any] = {"model": self._name, "fallback": False}

        async def call(headers: dict[str, Any], content: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            kw2 = dict(kw)
            forwarded = {k: v for k, v in headers.items() if k in FORWARD_HEADERS and isinstance(v, str)}
            if forwarded:
                kw2["model_settings"] = _with_headers(kw2.get("model_settings"), forwarded)
            try:
                if _fault_due():
                    raise InjectedFault("injected provider fault (FAULT_MODEL_CALLS)")
                resp = await self._inner.get_response(**kw2)
            except Exception as e:  # noqa: BLE001
                if self._fallback is None or not _worth_fallback(e):
                    raise
                err = f"{type(e).__name__}: {str(e)[:120]}"
                chosen.update(model=self._fallback_name, fallback=True, primary=self._name, primary_error=err)
                relay.mark_here("model_fallback", {"primary": self._name, "fallback": self._fallback_name, "error": err})
                if activity.in_activity():
                    activity.logger.warning("model fallback %s -> %s after %s", self._name, self._fallback_name, err)
                resp = await self._fallback.get_response(**kw2)
            return resp, {**chosen, **_response_summary(resp)}

        return await relay.model_call(self._name, _request_summary(kw), call)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.stream_response(*args, **kwargs)
# @@@SNIPEND


class RelayNimProvider(OpenAIProvider):
    def get_model(self, model_name: str | None) -> Model:
        name = model_name or DEFAULT_MODEL
        fallback_name = os.environ.get("NIM_FALLBACK_MODEL", DEFAULT_FALLBACK)
        fallback = super().get_model(fallback_name) if fallback_name and fallback_name != name else None
        return RelayModel(super().get_model(name), name, fallback, fallback_name if fallback else None)


def nim_model_provider() -> OpenAIProvider:
    set_tracing_disabled(True)  # the SDK's default trace exporter would post to api.openai.com
    return RelayNimProvider(
        openai_client=AsyncOpenAI(
            base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=os.environ.get("NVIDIA_API_KEY", "placeholder"),
            max_retries=0,
        ),
        use_responses=False,  # NIM serves chat completions, not the Responses API
    )


def nemotron_settings() -> ModelSettings:
    """Thinking off: Nemotron then answers a tool call in a few hundred tokens instead of reasoning past the token budget."""
    return ModelSettings(temperature=0.2, max_tokens=4096, parallel_tool_calls=False, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
