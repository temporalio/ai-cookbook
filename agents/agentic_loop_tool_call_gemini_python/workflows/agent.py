# ABOUTME: Main agentic loop workflow using Google Gemini.
# Orchestrates LLM calls and tool execution in a durable loop.

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities import gemini_chat
    from helpers import tool_helpers
    from tools import get_tools


def _serialize_tool(tool) -> dict:
    """Serialize a Tool object to a dict for passing to the activity."""
    function_declarations = []
    for fd in tool.function_declarations:
        fd_dict = {
            "name": fd.name,
            "description": fd.description,
        }
        if fd.parameters:
            properties = {}
            for prop_name, prop_schema in (fd.parameters.properties or {}).items():
                properties[prop_name] = {
                    "type": prop_schema.type.name if prop_schema.type else "STRING",
                    "description": prop_schema.description or "",
                }
            fd_dict["parameters"] = {
                "type": "OBJECT",
                "properties": properties,
                "required": list(fd.parameters.required or []),
            }
        function_declarations.append(fd_dict)
    return {"function_declarations": function_declarations}


@workflow.defn
class AgentWorkflow:
    """Agentic loop workflow that uses Gemini for LLM calls and executes tools."""

    @workflow.run
    async def run(self, input: str) -> str:
        """Run the agentic loop until the LLM produces a final response.

        Args:
            input: The user's initial message/query.

        Returns:
            The final text response from the LLM.
        """
        # Initialize conversation history with the user's message
        # Using Gemini's Content/Part structure (serialized for Temporal)
        contents = [{"role": "user", "parts": [{"text": input}]}]

        # Get tools and serialize for activity transport
        tools = [_serialize_tool(get_tools())]

        # The agentic loop
        while True:
            print(80 * "=")

            # Consult the LLM
            result = await workflow.execute_activity(
                gemini_chat.generate_content,
                gemini_chat.GeminiChatRequest(
                    model="gemini-flash-latest",
                    system_instruction=tool_helpers.HELPFUL_AGENT_SYSTEM_INSTRUCTIONS,
                    contents=contents,
                    tools=tools,
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )

            # Check if there are function calls to handle
            if result.function_calls:
                # Add the model's response (with function calls) to history
                contents.append({"role": "model", "parts": result.raw_parts})

                # Process each function call
                for function_call in result.function_calls:
                    tool_result = await self._handle_function_call(function_call)

                    # Add the function response to history
                    contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "function_response": {
                                        "name": function_call["name"],
                                        "response": {"result": tool_result},
                                    }
                                }
                            ],
                        }
                    )

            # If no function calls, we have a final response
            else:
                print(f"No tools chosen, responding with a message: {result.text}")
                return result.text

    async def _handle_function_call(self, function_call: dict) -> str:
        """Execute a tool via dynamic activity and return the result.

        Args:
            function_call: Dict containing 'name' and 'args' for the function.

        Returns:
            The string result from the tool execution.
        """
        tool_name = function_call["name"]
        tool_args = function_call.get("args", {})

        print(f"Making a tool call to {tool_name} with args: {tool_args}")

        result = await workflow.execute_activity(
            tool_name,
            tool_args,
            start_to_close_timeout=timedelta(seconds=30),
        )

        return result
