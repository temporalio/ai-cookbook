# ABOUTME: Main agentic loop workflow using Google Gemini.
# Orchestrates LLM calls and tool execution in a durable loop.

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    # Import pydantic internals early to avoid sandbox warnings
    import pydantic_core  # noqa: F401
    import annotated_types  # noqa: F401

    from google.genai import types

    from activities import gemini_chat
    from agent_config import prompts
    from tools import get_tools


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
        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=input)])
        ]

        # Get tools (cached - initialized by worker at startup)
        tools = [get_tools()]

        # The agentic loop
        while True:
            print(80 * "=")

            # Consult the LLM
            result = await workflow.execute_activity(
                gemini_chat.generate_content,
                gemini_chat.GeminiChatRequest(
                    model="gemini-2.5-flash",
                    system_instruction=prompts.SYSTEM_INSTRUCTIONS,
                    contents=contents,
                    tools=tools,
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )

            # Check if there are function calls to handle
            if result.function_calls:
                # Add the model's response (with function calls) to history
                contents.append(types.Content(role="model", parts=result.raw_parts))

                # Process each function call
                for function_call in result.function_calls:
                    tool_result = await self._handle_function_call(function_call)

                    # Add the function response to history
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_function_response(
                                    name=function_call["name"],
                                    response={"result": tool_result},
                                )
                            ],
                        )
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
