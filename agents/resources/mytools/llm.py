from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers only; avoid runtime import coupling
    from ..myprompts.provider import LLMProvider


def call_json_llm(
    prompt: str,
    client: Any,
    provider: "LLMProvider | str",
    model: str | None = None,
) -> str:
    """
    Call an LLM that returns a JSON-only text response.

    Args:
        prompt: Full text prompt to send.
        client: Provider-specific client instance.
        provider: Provider identifier, e.g. \"gemini\".
        model: Optional model name override.

    Returns:
        Raw text from the first candidate/choice, expected to be JSON.
    """

    # Normalize provider to a lowercase string, supporting Enum values
    if hasattr(provider, "value"):
        provider_name = str(getattr(provider, "value")).lower()
    else:
        provider_name = str(provider).lower()

    # Gemini implementation
    if provider_name == "gemini":
        from google import genai  # imported lazily to keep coupling minimal

        config = genai.types.GenerateContentConfig(
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(
            model=model or "gemini-2.5-pro",
            contents=prompt,
            config=config,
        )
        msg = resp.candidates[0].content
        part = msg.parts[0]
        txt = getattr(part, "text", None)
        if txt is None:
            txt = str(part)
        return txt
    # OpenAI implementation
    if provider_name == "openai":
        # Expect `client` to be an OpenAI client instance (from the `openai` package)
        # or the `openai` module itself, exposing `chat.completions.create`.
        response = client.chat.completions.create(
            model=model or "gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        # New OpenAI client returns a ChatCompletion with text content.
        return response.choices[0].message.content

    raise NotImplementedError(f"JSON LLM call not implemented for provider={provider!r}")


def call_text_llm(
    prompt: str,
    client: Any,
    provider: "LLMProvider | str",
    model: str | None = None,
) -> str:
    """
    Call an LLM that returns free-form text/Markdown.

    Args:
        prompt: Full text prompt to send.
        client: Provider-specific client instance.
        provider: Provider identifier, e.g. \"gemini\" or \"openai\".
        model: Optional model name override.

    Returns:
        Raw text from the first candidate/choice.
    """

    # Normalize provider to a lowercase string, supporting Enum values
    if hasattr(provider, "value"):
        provider_name = str(getattr(provider, "value")).lower()
    else:
        provider_name = str(provider).lower()

    if provider_name == "gemini":
        from google import genai  # imported lazily

        config = genai.types.GenerateContentConfig()
        resp = client.models.generate_content(
            model=model or "gemini-2.5-pro",
            contents=prompt,
            config=config,
        )
        msg = resp.candidates[0].content
        part = msg.parts[0]
        txt = getattr(part, "text", None)
        return txt if txt is not None else str(part)

    if provider_name == "openai":
        response = client.chat.completions.create(
            model=model or "gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    raise NotImplementedError(f"Text LLM call not implemented for provider={provider!r}")
