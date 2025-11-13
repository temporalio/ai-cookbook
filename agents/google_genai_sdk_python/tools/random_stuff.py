import random
import string
from pydantic import BaseModel, Field

# Pydantic models for tool parameters
class GetRandomNumberRequest(BaseModel):
    min_val: int = Field(description="The minimum value for the random number")
    max_val: int = Field(description="The maximum value for the random number")

class GetRandomStringRequest(BaseModel):
    length: int = Field(description="The length of the random string")

# Build tool definitions for Gemini API
GET_RANDOM_NUMBER_TOOL_GEMINI = {
    "function_declarations": [
        {
            "name": "get_random_number",
            "description": "Generate a random integer within a specified range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_val": {"type": "integer", "description": "The minimum value for the random number."},
                    "max_val": {"type": "integer", "description": "The maximum value for the random number."},
                },
                "required": ["min_val", "max_val"],
            },
        }
    ]
}

GET_RANDOM_STRING_TOOL_GEMINI = {
    "function_declarations": [
        {
            "name": "get_random_string",
            "description": "Generate a random string of a specified length.",
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {"type": "integer", "description": "The length of the random string."},
                },
                "required": ["length"],
            },
        }
    ]
}

# Tool functions
def get_random_number(req: GetRandomNumberRequest) -> int:
    return random.randint(req.min_val, req.max_val)

def get_random_string(req: GetRandomStringRequest) -> str:
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(req.length))
