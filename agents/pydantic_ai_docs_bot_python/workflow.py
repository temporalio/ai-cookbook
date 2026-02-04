"""Q&A workflow with signal-driven question handling."""

import asyncio
from pathlib import Path
from datetime import timedelta
from temporalio import workflow, activity

with workflow.unsafe.imports_passed_through():
    from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow, TemporalAgent
    from agents import dispatcher_agent, docs_agent, DispatcherDecision, DirectAnswer, SearchDocs


# ============================================================================
# Temporal Agents
# ============================================================================

temporal_dispatcher = TemporalAgent(dispatcher_agent)
temporal_docs_agent = TemporalAgent(docs_agent)


# ============================================================================
# Activities
# ============================================================================

@activity.defn
async def load_docs() -> dict[str, str]:
    """Load markdown files from docs/ directory.

    Returns:
        Dictionary of {filename: content}
    """
    docs_path = Path("docs")
    if not docs_path.exists():
        raise ValueError(
            "docs/ directory not found. Create it and add markdown files.\n"
            "Example: curl -o docs/workflows.md https://raw.githubusercontent.com/temporalio/documentation/main/docs/develop/python/core-application.md"
        )

    docs = {}
    for md_file in docs_path.glob("*.md"):
        if md_file.name != "README.md":
            docs[md_file.name] = md_file.read_text()

    if not docs:
        raise ValueError("No markdown files found in docs/")

    return docs


# ============================================================================
# Workflow
# ============================================================================

@workflow.defn
class QAWorkflow(PydanticAIWorkflow):
    """Q&A workflow that routes questions through multi-agent system."""

    __pydantic_ai_agents__ = [temporal_dispatcher, temporal_docs_agent]

    def __init__(self):
        self._questions = asyncio.Queue()
        self._answers: dict[str, str | dict] = {}
        self._docs: dict[str, str] = {}
        self._should_stop = False

    @workflow.run
    async def run(self) -> str:
        """Run the Q&A workflow.

        Returns:
            Status message
        """
        workflow.logger.info("Starting Q&A workflow")

        # Load documentation at startup
        self._docs = await workflow.execute_activity(
            load_docs,
            start_to_close_timeout=timedelta(seconds=30),
        )
        workflow.logger.info(f"Loaded {len(self._docs)} documentation files")

        # Process questions as they arrive
        while not self._should_stop:
            await workflow.wait_condition(
                lambda: not self._questions.empty() or self._should_stop
            )

            if self._should_stop:
                break

            question_text = await self._questions.get()
            workflow.logger.info(f"Processing: {question_text}")

            try:
                answer = await self._process_question(question_text)
                self._answers[question_text] = answer
            except Exception as e:
                workflow.logger.error(f"Error processing question: {e}")
                self._answers[question_text] = f"Error: {str(e)}"

        workflow.logger.info("Q&A session ending gracefully")
        return "Q&A session complete"

    @workflow.signal
    async def ask_question(self, question: str):
        """Signal to ask a question.

        Args:
            question: The question to ask
        """
        await self._questions.put(question)

    @workflow.signal
    def stop(self):
        """Signal to gracefully stop the workflow."""
        self._should_stop = True

    @workflow.query
    def get_answer(self, question: str) -> str | dict | None:
        """Query to get an answer.

        Args:
            question: The question to get answer for

        Returns:
            Answer or None if not found
        """
        return self._answers.get(question)

    async def _process_question(self, question: str) -> str | dict:
        """Process a question through the multi-agent system.

        Args:
            question: The question to process

        Returns:
            Answer string or dict
        """
        # Route through dispatcher
        dispatcher_result = await temporal_dispatcher.run(question)
        decision = dispatcher_result.output

        # Handle decision
        if isinstance(decision, DirectAnswer):
            workflow.logger.info(f"Direct answer (confidence: {decision.confidence})")
            return decision.answer

        elif isinstance(decision, SearchDocs):
            workflow.logger.info(f"Searching docs: {decision.query}")

            # Search documentation
            relevant_docs = {}
            for title, content in self._docs.items():
                if any(keyword.lower() in content.lower() for keyword in decision.keywords):
                    relevant_docs[title] = content

            if not relevant_docs:
                relevant_docs = self._docs

            # Format docs for the agent
            docs_context = "\n\n---\n\n".join([
                f"[{title}]\n{content[:500]}..."
                for title, content in list(relevant_docs.items())[:3]
            ])

            prompt = f"""Query: {decision.query}
Keywords: {', '.join(decision.keywords)}

Documentation:
{docs_context}

Based on these docs, provide a structured answer."""

            docs_result = await temporal_docs_agent.run(prompt)
            doc_answer = docs_result.output

            return {
                "answer": doc_answer.answer,
                "sources": doc_answer.sources,
                "confidence": doc_answer.confidence,
            }

        return "Unable to process question"
