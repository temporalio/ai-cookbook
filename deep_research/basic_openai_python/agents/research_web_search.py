from datetime import timedelta

from temporalio import workflow

from .config import EFFICIENT_PROCESSING_MODEL
from .shared import SearchQuery, SearchResult, with_today

with workflow.unsafe.imports_passed_through():
    from activities.invoke_model import InvokeModelRequest, invoke_model

WEB_SEARCH_INSTRUCTIONS = """
You are a web research specialist who finds and evaluates information from web sources.

CORE RESPONSIBILITIES:
1. Execute web searches using the web search tool
2. Prioritize authoritative sources: academic, government, established research organizations, prominent news outlets, primary sources
3. Extract key information relevant to the research question
4. Provide proper citations and assess reliability

APPROACH:
- Focus on information directly relevant to the research question
- Extract specific facts, data points, and evidence
- Note conflicting information and limitations
- Flag questionable or unverified claims

OUTPUT REQUIREMENTS:
- query: The search query that was executed
- sources: URLs and source descriptions consulted
- key_findings: Synthesized information relevant to research question (2-4 paragraphs)
- relevance_score: 0.0-1.0 assessment of how well results address the query
- citations: Formatted sources with URLs
"""


async def search_web(query: SearchQuery) -> SearchResult:
    search_input = f"""
Search Query: {query.query}
Query Rationale: {query.rationale}
Expected Information Type: {query.expected_info_type}
Priority Level: {query.priority}

Please search for information using the provided query and analyze the results according to the instructions.
"""
    result = await workflow.execute_activity(
        invoke_model,
        InvokeModelRequest(
            model=EFFICIENT_PROCESSING_MODEL,
            instructions=with_today(WEB_SEARCH_INSTRUCTIONS),
            input=search_input,
            response_format=SearchResult,
            tools=[{"type": "web_search"}],
        ),
        start_to_close_timeout=timedelta(seconds=300),
        summary="Searching web for information",
    )
    return result.response
