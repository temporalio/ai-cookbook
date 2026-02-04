"""Multi-agent system with dispatcher and documentation specialist."""

from dotenv import load_dotenv
load_dotenv()

import os
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import Literal

# ============================================================================
# Pydantic Models for Structured Outputs
# ============================================================================

class DirectAnswer(BaseModel):
    """Answer simple questions directly without searching docs."""
    type: Literal["direct"] = "direct"
    answer: str
    confidence: float  # 0.0-1.0


class SearchDocs(BaseModel):
    """Search documentation for detailed answer."""
    type: Literal["search"] = "search"
    query: str
    keywords: list[str]  # 2-5 keywords for search


# Union type for dispatcher decisions
DispatcherDecision = DirectAnswer | SearchDocs


class DocAnswer(BaseModel):
    """Structured answer from documentation search."""
    answer: str
    sources: list[str]  # List of doc titles where answer was found
    confidence: float


# ============================================================================
# PydanticAI Agents
# ============================================================================

# Auto-detect available API provider and select appropriate models
def _select_models() -> tuple[str, str]:
    """Select models based on available API keys.

    Returns:
        (dispatcher_model, docs_model)
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return ("anthropic:claude-haiku-4-5", "anthropic:claude-sonnet-4-5")
    elif os.getenv("OPENAI_API_KEY"):
        return ("openai:gpt-4o-mini", "openai:gpt-4o")
    elif os.getenv("GOOGLE_API_KEY"):
        return ("google:gemini-2.0-flash-exp", "google:gemini-2.0-flash-exp")
    else:
        raise ValueError(
            "No API key found. Set one of:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  GOOGLE_API_KEY=..."
        )

_dispatcher_model, _docs_model = _select_models()

# Dispatcher Agent - routes questions to appropriate handler
dispatcher_agent = Agent(
    _dispatcher_model,  # Fast, cheap model for routing
    output_type=DispatcherDecision,
    name='dispatcher',  # Required for Temporal
    system_prompt="""You are a question routing agent for a documentation Q&A system.

Your job is to decide how to handle each question:

1. DirectAnswer: Use for simple, general questions that don't need documentation.
   - Example: "What is Temporal?" → Direct answer
   - Confidence should be 0.8+ if you're sure

2. SearchDocs: Use for questions requiring specific documentation details.
   - Example: "How do I retry an activity?" → Search docs
   - Extract 2-5 keywords for search

Be conservative: when in doubt, search docs for accurate answers.""",
)


# Documentation Agent - searches docs and provides detailed answers
docs_agent = Agent(
    _docs_model,  # More capable model for comprehension
    output_type=DocAnswer,
    name='docs_search',  # Required for Temporal
    system_prompt="""You are a documentation search agent.

Given documentation chunks and a search query, your job is to:
1. Analyze the documentation for relevant information
2. Formulate a clear, concise answer
3. List which documentation sources you used
4. Provide a confidence score (0.7-1.0)

Be direct and practical in your answers.""",
)


async def route_question(question: str) -> DispatcherDecision:
    """Route a question through the dispatcher agent.

    Args:
        question: The question to route

    Returns:
        DispatcherDecision (either DirectAnswer or SearchDocs)
    """
    result = await dispatcher_agent.run(question)
    return result.output


async def search_documentation(query: str, keywords: list[str], docs: dict[str, str]) -> DocAnswer:
    """Search documentation and generate answer.

    Args:
        query: The search query
        keywords: Keywords to help find relevant docs
        docs: Dictionary of {title: content} documentation

    Returns:
        DocAnswer with answer, sources, and confidence
    """
    # Simple keyword-based search
    relevant_docs = {}
    for title, content in docs.items():
        # Check if any keyword appears in the doc
        if any(keyword.lower() in content.lower() for keyword in keywords):
            relevant_docs[title] = content

    if not relevant_docs:
        relevant_docs = docs  # Fall back to all docs

    # Format docs for the agent
    docs_context = "\n\n---\n\n".join([
        f"[{title}]\n{content[:500]}..."  # First 500 chars of each doc
        for title, content in list(relevant_docs.items())[:3]  # Max 3 docs
    ])

    prompt = f"""Query: {query}
Keywords: {', '.join(keywords)}

Documentation:
{docs_context}

Based on these docs, provide a structured answer."""

    result = await docs_agent.run(prompt)
    return result.output
