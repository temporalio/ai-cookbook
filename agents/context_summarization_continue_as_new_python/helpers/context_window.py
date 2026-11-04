from __future__ import annotations

from typing import Any

# A rough characters-per-token ratio. Tokenizing for real would need a network
# call or a model-specific library, neither of which is allowed in workflow code,
# so the window estimates from character length instead.
DEFAULT_CHARS_PER_TOKEN = 4

COMPACTION_NOTE = (
    "Earlier turns in this conversation were dropped to fit the context window. "
    "Rely on the recent turns below; ask the user to repeat anything missing."
)


def estimate_tokens(
    messages: list[dict[str, Any]],
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Estimate the token cost of a message list from its character length."""
    chars = 0
    for message in messages:
        content = message["content"]
        chars += len(content if isinstance(content, str) else str(content))
    return chars // chars_per_token + 1


def window_messages(
    messages: list[dict[str, Any]],
    *,
    max_recent: int,
    max_context_tokens: int,
    preserve_initial: bool = True,
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
) -> tuple[list[dict[str, Any]], int]:
    """Choose which messages to send to the model, deterministically.

    Keeps at most ``max_recent`` messages, optionally pins the initial user
    message (it usually states the task), then sheds the oldest kept turns until
    the estimate fits ``max_context_tokens``. The result keeps user/assistant
    turns alternating and starting with a user turn, so it is a valid provider
    request. Returns the selected messages and the number of messages dropped.

    This runs in workflow code, so it is pure and deterministic: no clocks, no
    randomness, no I/O.
    """
    if not messages:
        return [], 0
    within_count = len(messages) <= max_recent
    within_budget = estimate_tokens(messages, chars_per_token) <= max_context_tokens
    if within_count and within_budget:
        return list(messages), 0

    head = [messages[0]] if preserve_initial else []
    tail_start = max(len(head), len(messages) - (max_recent - len(head)))
    tail = messages[tail_start:]
    tail = _trim_front_for_valid_request(head, tail)

    while tail and estimate_tokens(head + tail, chars_per_token) > max_context_tokens:
        tail = _trim_front_for_valid_request(head, tail[1:])

    selected = head + tail
    return selected, len(messages) - len(selected)


def _trim_front_for_valid_request(
    head: list[dict[str, Any]],
    tail: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop a leading turn so head+tail alternates and starts with a user turn.

    A pinned initial user message must be followed by an assistant turn; with no
    pinned head, the request itself must start with a user turn.
    """
    if head:
        if tail and tail[0]["role"] == "user":
            return tail[1:]
        return tail
    if tail and tail[0]["role"] != "user":
        return tail[1:]
    return tail
