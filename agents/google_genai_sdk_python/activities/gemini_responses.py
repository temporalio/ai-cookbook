from temporalio import activity
import google.generativeai as genai
from dataclasses import dataclass
from typing import Any, Optional
import os

# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class GeminiResponsesRequest:
    model: str
    instructions: str
    input: list[dict[str, Any]]
    tools: list[Any]

# Simple wrapper to convert Gemini tool format to what GenerativeModel expects

@dataclass
class GeminiResponse:
    """Wrapper to match response structure"""
    output: list
    output_text: str

@dataclass
class GeminiOutputItem:
    """Wrapper for output items to match Gemini's output structure"""
    type: str
    name: Optional[str] = None
    call_id: Optional[str] = None
    arguments: Optional[dict] = None
    content: Optional[str] = None

@activity.defn
async def create(request: GeminiResponsesRequest) -> GeminiResponse:
    """
    Invoke Gemini API with conversation history and tools.
    Returns a response structure compatible with the agentic loop workflow.
    """
    # Configure Gemini API
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Create model with system instructions and tools
    model = genai.GenerativeModel(
        request.model,
        system_instruction=request.instructions,
        tools=request.tools
    )

    # Convert input list to Gemini's expected format
    # The input list follows OpenAI format with type, role, content structure
    history = []

    # Check if the last item is a function_call_output - if so, include it in history
    last_item = request.input[-1]
    is_continuing_after_tool = last_item.get("type") == "function_call_output"

    # If we're continuing after a tool call, include all items in history
    # Otherwise, all but the last item go into history
    items_for_history = request.input if is_continuing_after_tool else request.input[:-1]

    for item in items_for_history:
        if item.get("type") == "message":
            history.append({
                "role": item["role"],
                "parts": [item["content"]]
            })
        elif item.get("type") == "function_call":
            # Model's tool call
            history.append({
                "role": "model",
                "parts": [{
                    "function_call": {
                        "name": item["name"],
                        "args": dict(item["arguments"])
                    }
                }]
            })
        elif item.get("type") == "function_call_output":
            # Tool response
            history.append({
                "role": "function",
                "parts": [{
                    "function_response": {
                        "name": item["call_id"],
                        "response": {"result": item["output"]}
                    }
                }]
            })

    # Start chat with history
    chat = model.start_chat(history=history)

    # If continuing after a tool call, send continuation prompt to get model's response
    # Otherwise, send the last user message
    if is_continuing_after_tool:
        # After a function response, we need to prompt Gemini to continue
        # Use a simple continuation prompt
        response = await chat.send_message_async("Please continue and provide your response based on the tool results.")
    else:
        prompt_content = last_item.get("content", "")
        response = await chat.send_message_async(prompt_content)

    # Parse response and convert to output structure
    output = []
    output_text = ""

    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                # Tool call detected
                output.append(GeminiOutputItem(
                    type="function_call",
                    name=part.function_call.name,
                    call_id=part.function_call.name,  # Use name as call_id
                    arguments=dict(part.function_call.args)
                ))
            elif hasattr(part, 'text') and part.text:
                # Text response
                output_text += part.text

    # If no tool calls, add text as message output
    if not any(item.type == "function_call" for item in output):
        output.append(GeminiOutputItem(
            type="message",
            content=output_text
        ))

    return GeminiResponse(output=output, output_text=output_text)
