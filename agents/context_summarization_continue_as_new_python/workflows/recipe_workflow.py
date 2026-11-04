from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel, Field

    from activities.llm_call import CallLlmRequest, call_llm
    from helpers.context_window import COMPACTION_NOTE, window_messages

DEFAULT_SYSTEM = "You are a helpful assistant. Keep your replies concise."


class SummarizationConfig(BaseModel):
    """Knobs for the model call and the context/continuation policy."""

    model: str = "claude-sonnet-4-6"
    system: str = DEFAULT_SYSTEM
    max_tokens: int = 1024
    # Context-window policy applied before every model call.
    max_recent_messages: int = 12
    max_context_tokens: int = 8_000
    # Continuation policy. 0 means "only continue-as-new when Temporal suggests
    # it" (the production default); a positive value forces a handoff every N
    # turns, which makes the pattern easy to demonstrate and test.
    continue_as_new_after_turns: int = 0


class AgentState(BaseModel):
    """The compacted snapshot carried across a continue-as-new boundary."""

    messages: list[dict[str, Any]]
    turns_completed: int = 0
    compactions: int = 0


class AgentInput(BaseModel):
    """Workflow input: the remaining conversation turns plus carried state."""

    prompts: list[str]
    config: SummarizationConfig = Field(default_factory=SummarizationConfig)
    agent_state: AgentState | None = None


class AgentResult(BaseModel):
    final_message: str
    total_turns: int
    compactions: int
    final_context_messages: int


@workflow.defn
class SummarizingAgentWorkflow:
    @workflow.run
    async def run(self, agent_input: AgentInput) -> AgentResult:
        if not agent_input.prompts:
            raise ApplicationError(
                "AgentInput.prompts must contain at least one turn.",
                non_retryable=True,
            )

        config = agent_input.config

        # Resume from the compacted snapshot when this run is a continuation;
        # otherwise start an empty conversation.
        if agent_input.agent_state is not None:
            messages: list[dict[str, Any]] = list(agent_input.agent_state.messages)
            turns_completed = agent_input.agent_state.turns_completed
            compactions = agent_input.agent_state.compactions
        else:
            messages = []
            turns_completed = 0
            compactions = 0

        remaining = list(agent_input.prompts)
        turns_this_run = 0
        final_message = ""

        while remaining:
            messages.append({"role": "user", "content": remaining.pop(0)})

            # Window the conversation down to a bounded, valid request before the
            # model ever sees it. The full history stays in `messages`; only the
            # windowed view is sent.
            model_messages, dropped = window_messages(
                messages,
                max_recent=config.max_recent_messages,
                max_context_tokens=config.max_context_tokens,
            )
            system = config.system
            if dropped:
                system = f"{config.system}\n\n[Context note] {COMPACTION_NOTE}"

            reply = await workflow.execute_activity(
                call_llm,
                CallLlmRequest(
                    model=config.model,
                    system=system,
                    messages=model_messages,
                    max_tokens=config.max_tokens,
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )
            messages.append({"role": "assistant", "content": reply})
            final_message = reply
            turns_completed += 1
            turns_this_run += 1

            # Hand off to a fresh run before the history grows unbounded, carrying
            # a compacted snapshot so the next run keeps the conversation going
            # with a clean, small history.
            if remaining and self._should_continue_as_new(turns_this_run, config):
                compacted, _ = window_messages(
                    messages,
                    max_recent=config.max_recent_messages,
                    max_context_tokens=config.max_context_tokens,
                )
                workflow.continue_as_new(
                    AgentInput(
                        prompts=remaining,
                        config=config,
                        agent_state=AgentState(
                            messages=compacted,
                            turns_completed=turns_completed,
                            compactions=compactions + 1,
                        ),
                    )
                )

        return AgentResult(
            final_message=final_message,
            total_turns=turns_completed,
            compactions=compactions,
            final_context_messages=len(messages),
        )

    def _should_continue_as_new(self, turns_this_run: int, config: SummarizationConfig) -> bool:
        if config.continue_as_new_after_turns and turns_this_run >= config.continue_as_new_after_turns:
            return True
        # Temporal raises this flag as workflow history approaches the size where
        # continue-as-new is the right move.
        return workflow.info().is_continue_as_new_suggested()
