from pydantic import BaseModel
from typing import Optional, List

from .models import (
    SystemPrompt,
    GoalPrompt,
    InitialPlanPrompt,
    TemplatePrompt,
    UserPrompt,
    BasePrompt,
)
from .provider import LLMProvider
from .assembly import PromptAssembly


class PromptBundle(BaseModel):
    """
    Bundle of prompts that define an agent's behavior.
    This becomes the workflow's input.
    """

    system: Optional[SystemPrompt] = None
    goal: Optional[GoalPrompt] = None
    plan: Optional[InitialPlanPrompt] = None
    context_prompts: List[BasePrompt] = []
    instruction_prompts: List[BasePrompt] = []
    provider: LLMProvider = LLMProvider.GEMINI

    def assemble(self) -> List[dict]:
        prompts = []

        if self.system:
            prompts.append(self.system)
        if self.goal:
            prompts.append(self.goal)
        if self.plan:
            prompts.append(self.plan)

        prompts.extend(self.context_prompts)
        prompts.extend(self.instruction_prompts)

        return PromptAssembly(prompts=prompts).build(provider=self.provider)