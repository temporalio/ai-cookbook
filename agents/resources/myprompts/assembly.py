from typing import List, Dict
from pydantic import BaseModel
from .models import BasePrompt, MessageBlock
from .provider import LLMProvider


class PromptAssembly(BaseModel):
    """
    Assemble a sequence of BasePrompt or MessageBlock instances
    into a provider-appropriate message payload.

    For Gemini → returns a `contents=` list.
    For OpenAI → returns a `messages=` list.
    """

    prompts: List[BasePrompt]

    def build(self, provider: LLMProvider = LLMProvider.GEMINI) -> List[Dict[str, str]]:
        """
        Build a message list for a given provider.

        Args:
            provider (LLMProvider): Enum specifying LLM backend (gemini / openai)

        Returns:
            List[Dict[str, str]]: A list of role/content message dicts
                                  formatted for the specific provider.
        """
        messages: List[Dict[str, str]] = []

        for prompt in self.prompts:
            # Each prompt knows how to format itself for a given provider
            messages.extend(prompt.to_messages(provider=provider))

        return messages
