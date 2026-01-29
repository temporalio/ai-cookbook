"""
Simple test script for Claude API with tool calling.

This is a standalone test that doesn't use Temporal workflows.
Use this to verify your Claude API setup before running the full workflow.

Usage:
    export ANTHROPIC_API_KEY='your-api-key'
    python claude_test.py
"""

import os
from anthropic import Anthropic

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Define a simple tool
tool_definition = {
    "name": "add_numbers",
    "description": "Adds two integers together",
    "input_schema": {
        "type": "object",
        "properties": {
            "left": {
                "type": "integer",
                "description": "The first integer to add"
            },
            "right": {
                "type": "integer",
                "description": "The second integer to add"
            }
        },
        "required": ["left", "right"]
    }
}

# First call - Claude will decide to use the tool
print("=" * 80)
print("Calling Claude with tool...")
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[tool_definition],
    messages=[
        {
            "role": "user",
            "content": "What is 9 + 10?",
        }
    ],
)

print("\nClaude's response:")
for block in message.content:
    if block.type == "text":
        print(f"Text: {block.text}")
    elif block.type == "tool_use":
        print(f"Tool call: {block.name}")
        print(f"Arguments: {block.input}")

        # Simulate tool execution
        result = block.input["left"] + block.input["right"]
        print(f"Tool result: {result}")

        # Send result back to Claude
        print("\n" + "=" * 80)
        print("Sending tool result back to Claude...")

        final_message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[tool_definition],
            messages=[
                {"role": "user", "content": "What is 9 + 10?"},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        }
                    ]
                }
            ],
        )

        print("\nClaude's final response:")
        for final_block in final_message.content:
            if final_block.type == "text":
                print(f"{final_block.text}")