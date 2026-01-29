from .registry import register_tool, TOOL_REGISTRY, DISPATCH_TABLE
from .schemas import build_gemini_schema, build_openai_tools


def load_tool_modules(*module_paths: str) -> None:
    """
    Import one or more modules by dotted path so that any
    `@tool`-decorated functions they contain are registered.

    Example:
        load_tool_modules("workflows.gemini_research_agent.company_research_tools")
    """
    import importlib

    for path in module_paths:
        importlib.import_module(path)


# Backwards-compatible aliases used by workflow activities
TOOL_DISPATCH = DISPATCH_TABLE
TOOL_SCHEMAS = [build_gemini_schema()]
OPENAI_TOOLS = build_openai_tools()
