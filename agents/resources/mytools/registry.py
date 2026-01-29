import inspect
from typing import get_type_hints, get_origin, get_args
from pydantic import BaseModel

TOOL_REGISTRY = []
DISPATCH_TABLE = {}

def build_schema_from_pydantic_model(model: type[BaseModel]):
    """Return JSON schema for Gemini from a Pydantic model."""
    schema = model.model_json_schema()

    props = {}
    required = schema.get("required", [])

    for name, prop in schema["properties"].items():
        ptype = prop.get("type", "string")

        # Normalize to Gemini-compatible JSON schema
        if ptype == "array":
            # Gemini requires "items" for array types
            item_schema = prop.get("items", {})
            item_type = item_schema.get("type", "string")
            props[name] = {
                "type": "array",
                "items": {"type": item_type},
            }
        else:
            props[name] = {"type": ptype}

    return {
        "type": "object",
        "properties": props,
        "required": required
    }


def build_schema_from_function(func):
    """
    Build a function schema for Gemini.
    Supports:
    - Plain Python type hints
    - Pydantic model argument
    """

    sig = inspect.signature(func)
    hints = get_type_hints(func)

    # Case 1: Pydantic model as the ONLY argument
    if len(sig.parameters) == 1:
        (param_name, param), = sig.parameters.items()

        annotation = hints.get(param_name)

        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            model = annotation
            params_schema = build_schema_from_pydantic_model(model)
            return {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": params_schema
            }

    # Case 2: Standard type-hinted Python function
    props = {}
    required = []

    for name, param in sig.parameters.items():
        if name in hints:
            py_type = hints[name]
            if py_type == str:
                props[name] = {"type": "string"}
            elif py_type == int:
                props[name] = {"type": "integer"}
            elif py_type == float:
                props[name] = {"type": "number"}
            elif py_type == bool:
                props[name] = {"type": "boolean"}
            elif inspect.isclass(py_type) and issubclass(py_type, BaseModel):
                # Nested Pydantic model (rare but supported)
                props[name] = build_schema_from_pydantic_model(py_type)
            else:
                props[name] = {"type": "string"}
        else:
            props[name] = {"type": "string"}

        required.append(name)

    return {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required
        }
    }


def register_tool(func):
    """Register a tool and build schema."""
    schema = build_schema_from_function(func)
    TOOL_REGISTRY.append(schema)
    DISPATCH_TABLE[func.__name__] = func
    return func
