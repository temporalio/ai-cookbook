from typing import Dict
from .models import BasePrompt

PROMPT_LIBRARY: Dict[str, BasePrompt] = {}


def register_prompt(name: str, prompt: BasePrompt):
    PROMPT_LIBRARY[name] = prompt
    return prompt


def get_prompt(name: str) -> BasePrompt:
    return PROMPT_LIBRARY[name]

