from collections.abc import Callable
from typing import Any

# The supervisor's one tool: delegate a focused sub-task to a subagent.
DELEGATE_TO_SUBAGENT = "delegate_to_subagent"


# Two trivial, deterministic tools for the subagent. They take a single string and
# return a string, so they are safe to run inline inside a workflow (no I/O, no clock,
# no randomness). A real subagent would call these through Activities instead.
def word_count(text: str) -> str:
    return str(len(text.split()))


def to_upper(text: str) -> str:
    return text.upper()


SUBAGENT_TOOLS: dict[str, Callable[[str], str]] = {
    "word_count": word_count,
    "to_upper": to_upper,
}


def delegate_tool_schema() -> dict[str, Any]:
    # The supervisor describes delegation to Claude as a single tool. `tool_names`
    # lets the model pick which subagent tools to grant for the sub-task.
    return {
        "name": DELEGATE_TO_SUBAGENT,
        "description": (
            "Delegate a self-contained sub-task to a focused subagent. The subagent "
            "runs in its own context with a restricted toolset and returns a short result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The sub-task for the subagent."},
                "tool_names": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(SUBAGENT_TOOLS)},
                    "description": "Subagent tools to grant for this sub-task.",
                },
            },
            "required": ["task"],
        },
    }


def subagent_tool_schemas(tool_names: list[str]) -> list[dict[str, Any]]:
    # Only expose the granted tools to the subagent. The delegate tool is never
    # in this set, so a subagent cannot recursively spawn more subagents.
    schemas: list[dict[str, Any]] = []
    for name in tool_names:
        if name in SUBAGENT_TOOLS:
            schemas.append(
                {
                    "name": name,
                    "description": f"Apply {name} to a string.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            )
    return schemas
