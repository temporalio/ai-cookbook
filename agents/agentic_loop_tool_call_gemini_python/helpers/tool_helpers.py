# ABOUTME: Helper functions for creating Gemini tool definitions from Pydantic models.
# Also contains the system instructions for the agent.

from pydantic import BaseModel
from google.genai import types


def gemini_tool_from_model(
    name: str, description: str, model: type[BaseModel] | None
) -> types.FunctionDeclaration:
    """Create a Gemini FunctionDeclaration from a Pydantic model.

    Args:
        name: The name of the function/tool.
        description: A description of what the tool does.
        model: A Pydantic model class describing the parameters, or None for no parameters.

    Returns:
        A FunctionDeclaration that can be used with the Gemini API.
    """
    if model is None:
        # Tool with no parameters
        return types.FunctionDeclaration(
            name=name,
            description=description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
                required=[],
            ),
        )

    # Build parameters from Pydantic model schema
    schema = model.model_json_schema()
    properties = {}

    for prop_name, prop_info in schema.get("properties", {}).items():
        prop_type = prop_info.get("type", "string")
        prop_desc = prop_info.get("description", "")

        # Map JSON schema types to Gemini types
        type_mapping = {
            "string": types.Type.STRING,
            "integer": types.Type.INTEGER,
            "number": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN,
            "array": types.Type.ARRAY,
            "object": types.Type.OBJECT,
        }

        gemini_type = type_mapping.get(prop_type, types.Type.STRING)
        properties[prop_name] = types.Schema(type=gemini_type, description=prop_desc)

    required = schema.get("required", [])

    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties=properties,
            required=required,
        ),
    )


HELPFUL_AGENT_SYSTEM_INSTRUCTIONS = """
You are a helpful agent that can use tools to help the user.
You will be given an input from the user and a list of tools to use.
You may or may not need to use the tools to satisfy the user ask.
If no tools are needed, respond in haikus.
"""
