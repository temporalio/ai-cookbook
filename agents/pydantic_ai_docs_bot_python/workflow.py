"""Documentation Q&A workflow demonstrating agentic loop with PydanticAI."""

from pathlib import Path
from datetime import timedelta
from temporalio import workflow, activity

with workflow.unsafe.imports_passed_through():
    from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow, TemporalAgent
    from agents import documentation_agent, DocsContext


# ============================================================================
# Temporal Agent
# ============================================================================

temporal_agent = TemporalAgent(documentation_agent)


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
class DocumentationAgent(PydanticAIWorkflow):
    """Documentation Q&A agent demonstrating autonomous tool calling."""

    __pydantic_ai_agents__ = [temporal_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run the documentation agent with a single prompt.

        The agent will autonomously call tools as needed to answer the question.

        Args:
            prompt: User question or request

        Returns:
            Agent's final answer
        """
        # Load documentation
        docs = await workflow.execute_activity(
            load_docs,
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Run agent - it will autonomously call tools as needed
        result = await temporal_agent.run(
            prompt,
            deps=DocsContext(docs=docs)
        )

        return result.output
