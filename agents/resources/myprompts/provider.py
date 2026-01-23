from enum import Enum

class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    # Future-proofing:
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    MISTRAL = "mistral"