# myprompts/history.py

from pydantic import BaseModel
from typing import List
from .models import BasePrompt
from .assembly import PromptAssembly
from .provider import LLMProvider


class HistoryEntry(BaseModel):
    prompt: BasePrompt


class PromptHistory(BaseModel):
    entries: List[HistoryEntry] = []

    def add(self, prompt: BasePrompt):
        self.entries.append(HistoryEntry(prompt=prompt))

    def to_messages(self, provider: LLMProvider):
        prompts = [entry.prompt for entry in self.entries]
        return PromptAssembly(prompts=prompts).build(provider=provider)
