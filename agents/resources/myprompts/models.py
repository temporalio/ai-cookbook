from typing import Dict, Any, List
from pydantic import BaseModel


# -------------------------
# Base Prompt Model
# -------------------------
OPENAI_ROLES = {"system", "user", "assistant", "tool"}
GEMINI_ALLOWED_ROLES = {"user", "model"}
from .provider import LLMProvider



def map_role_to_openai(role: str) -> str:
    role = role.lower()
    return role if role in OPENAI_ROLES else "user"


def map_role_to_gemini(role: str) -> str:
    role = role.lower()

    if role == "system":
        return "user"      # Gemini has no system role
    if role == "assistant":
        return "model"
    if role == "tool":
        return "model"

    return "user"

# ============================================================
# Base Prompt
# ============================================================

class BasePrompt(BaseModel):
    role: str = "user"
    text: str

    def to_gemini(self) -> List[Dict[str, str]]:
        """
        Return a Gemini `Content`-compatible structure.
        """
        return [{
            "role": map_role_to_gemini(self.role),
            "parts": [{"text": self.text}],
        }]

    def to_openai(self) -> List[Dict[str, str]]:
        return [{
            "role": map_role_to_openai(self.role),
            "content": self.text
        }]

    def to_messages(self, provider: LLMProvider = LLMProvider.GEMINI) -> List[Dict[str, str]]:
        if provider == LLMProvider.GEMINI:
            return self.to_gemini()
        if provider == LLMProvider.OPENAI:
            return self.to_openai()
        raise ValueError(f"Unknown provider: {provider}")


# -------------------------
# Specific Prompt Types
# -------------------------

class SystemPrompt(BasePrompt):
    role: str = "system"


class UserPrompt(BasePrompt):
    role: str = "user"


class TaskPrompt(BasePrompt):
    """Represents a high-level user task"""
    role: str = "user"

class InitialPlanPrompt(BasePrompt):
    role: str = "user"

# -------------------------
# MessageBlock for multi-part prompts
# -------------------------

class MessageBlock(BaseModel):
    """
    Represents multi-part content prompts, such as context + instructions.
    """
    role: str
    content: List[str]

    def to_messages(self) -> List[Dict[str, str]]:
        return [{"role": self.role, "content": part} for part in self.content]


# -------------------------
# Template Prompt
# -------------------------
class TemplatePrompt(BasePrompt):
    variables: Dict[str, Any] = {}

    def render(self):
        text = self.text
        for key, val in self.variables.items():
            text = text.replace(f"{{{{{key}}}}}", str(val))
        return TemplatePrompt(role=self.role, text=text, variables=self.variables)

    def to_messages(self, provider: LLMProvider = LLMProvider.GEMINI):
        if provider == LLMProvider.GEMINI:
            return self.to_gemini()
        elif provider == LLMProvider.OPENAI:
            return self.to_openai()
        else:
            raise ValueError(f"Unknown provider: {provider}")


class MessageBlock(BaseModel):
    role: str
    content: List[str]

    def to_messages(self, provider: LLMProvider = LLMProvider.GEMINI):
        if provider == LLMProvider.GEMINI:
            return self.to_gemini()
        elif provider == LLMProvider.OPENAI:
            return self.to_openai()
        else:
            raise ValueError(f"Unknown provider: {provider}")

