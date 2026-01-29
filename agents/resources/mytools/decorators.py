from .registry import register_tool

def tool(func):
    """Decorator that registers a tool function."""
    register_tool(func)
    return func
