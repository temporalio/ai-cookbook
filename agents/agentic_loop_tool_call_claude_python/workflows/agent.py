from temporalio import workflow
from datetime import timedelta
import json

with workflow.unsafe.imports_passed_through():
    from tools import get_tools
    from helpers import tool_helpers
    from activities import claude_responses

@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, input: str) -> str:
        """
        Agentic loop using Claude (Anthropic) API.
        
        Key differences from OpenAI:
        - Claude uses 'messages' array with 'role' and 'content'
        - Tool calls come back as content blocks with type 'tool_use'
        - Tool results are sent back as messages with 'tool_result' content blocks
        """
        
        # Initialize messages list with user input
        messages = [{"role": "user", "content": input}]

        # The agentic loop
        while True:
            print(80 * "=")
                
            # Consult Claude
            result = await workflow.execute_activity(
                claude_responses.create,
                claude_responses.ClaudeResponsesRequest(
                    model="claude-sonnet-4-20250514",
                    system=tool_helpers.HELPFUL_AGENT_SYSTEM_INSTRUCTIONS,
                    messages=messages,
                    tools=get_tools(),
                    max_tokens=4096,
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )

            # Claude returns content blocks - check if any are tool_use
            tool_use_blocks = [block for block in result.content if block.type == "tool_use"]
            
            if tool_use_blocks:
                # We have tool calls to handle
                # First, add the assistant's response to messages
                # Convert content blocks to dictionaries for serialization
                assistant_content = []
                for block in result.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                
                messages.append({"role": "assistant", "content": assistant_content})
                
                # Execute all tool calls and collect results
                tool_results = []
                for block in tool_use_blocks:
                    print(f"[Agent] Tool call: {block.name}({block.input})")
                    
                    # Execute the tool
                    tool_result = await self._execute_tool(block.name, block.input)
                    
                    print(f"[Agent] Tool result: {tool_result}")
                    
                    # Add tool result in Claude's expected format
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(tool_result)
                    })
                
                # Add tool results as a user message
                messages.append({"role": "user", "content": tool_results})
            else:
                # No tool calls - extract the text response and return
                text_blocks = [block for block in result.content if block.type == "text"]
                if text_blocks:
                    response_text = text_blocks[0].text
                    print(f"[Agent] Final response: {response_text}")
                    return response_text
                else:
                    return "No text response from Claude"

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a tool dynamically.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters
        """
        # Execute dynamic activity with the tool name and arguments
        result = await workflow.execute_activity(
            tool_name,
            tool_input,
            start_to_close_timeout=timedelta(seconds=30),
        )
        return result

