from temporalio import activity
import google.generativeai as genai
from dataclasses import dataclass
from typing import Any
import os

# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class GeminiResponsesRequest:
    model: str
    instructions: str
    history: list[dict[str, Any]]
    prompt: str
    tools: list[Any]

def serialize_response(response: Any) -> dict[str, Any]:
    """
    Convert Gemini API response to serializable format.
    Extracts function calls and text from response parts.
    """
    serialized_parts = []
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            part_dict = {}
            if hasattr(part, 'function_call') and part.function_call:
                part_dict["function_call"] = {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                }
            elif hasattr(part, 'text') and part.text:
                part_dict["text"] = part.text
            serialized_parts.append(part_dict)

    return {"parts": serialized_parts}

@activity.defn
async def create(request: GeminiResponsesRequest) -> dict[str, Any]:
    """
    Invoke Gemini API with pre-built conversation history and tools.
    Returns the raw response from chat.send_message_async() in serializable format.
    """
    # Configure Gemini API
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Create model with system instructions and tools
    model = genai.GenerativeModel(
        request.model,
        system_instruction=request.instructions,
        tools=request.tools
    )

    # Start chat with pre-built history
    chat = model.start_chat(history=request.history)

    # Send the prompt
    response = await chat.send_message_async(request.prompt)

    # Serialize and return response
    return serialize_response(response)
