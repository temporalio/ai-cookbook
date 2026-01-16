# random_stuff.py
# Example of a simple tool with no parameters

from typing import Any
import random
from pydantic import BaseModel, Field
from helpers import tool_helpers

# For tools without parameters, we can pass None as the model
RANDOM_NUMBER_TOOL_CLAUDE: dict[str, Any] = tool_helpers.claude_tool_from_model(
    "get_random_number",
    "Get a random number between 1 and 100. Use this when the user asks for a random number or wants to play a guessing game.",
    None
)

async def get_random_number() -> str:
    """Generate a random number between 1 and 100."""
    number = random.randint(1, 100)
    return str(number)


# Example of a tool with parameters
class GenerateRandomTextRequest(BaseModel):
    length: int = Field(description="The length of the random text to generate (number of words)", ge=1, le=100)

RANDOM_TEXT_TOOL_CLAUDE: dict[str, Any] = tool_helpers.claude_tool_from_model(
    "generate_random_text",
    "Generate random Lorem Ipsum text with a specified number of words.",
    GenerateRandomTextRequest
)

async def generate_random_text(req: GenerateRandomTextRequest) -> str:
    """Generate random Lorem Ipsum text."""
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", 
             "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", 
             "et", "dolore", "magna", "aliqua"]
    
    result = []
    for _ in range(req.length):
        result.append(random.choice(words))
    
    return " ".join(result).capitalize() + "."

