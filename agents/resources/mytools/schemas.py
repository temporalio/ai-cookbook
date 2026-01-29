from .registry import TOOL_REGISTRY


def build_gemini_schema():
    return {
        "function_declarations": TOOL_REGISTRY
    }


def build_openai_tools():
    """
    Return tools in OpenAI-compatible format:
    [
      {
        "type": "function",
        "function": {
          "name": ...,
          "description": ...,
          "parameters": {...}
        }
      },
      ...
    ]
    """
    return [
        {
            "type": "function",
            "function": schema,
        }
        for schema in TOOL_REGISTRY
    ]
