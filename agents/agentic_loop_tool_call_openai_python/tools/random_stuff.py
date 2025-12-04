# random_stuff.py

from typing import Any
from helpers import tool_helpers
import random

# Build the tool definition for the OpenAI Responses API. 
RANDOM_NUMBER_TOOL_OAI: dict[str, Any] = tool_helpers.oai_responses_tool_from_model(
    "get_random_number",
    "Get a random number between 0 and 100.",
    None)

# The function
async def get_random_number() -> str:
    """Get a random number between 0 and 100.
    """
    data = random.randint(0, 100)
    return str(data)
