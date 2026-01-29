from myprompts import *
from myprompts.provider import LLMProvider

company = "Google"

SYSTEM_PROMPT = """
You are an expert Competitive Analysis Agent.
Given a company name,
validate it using LLM knowledge,
determine its sector,
identify top 3 competitors,
gather real-time strategy data using tools,
analyze their strategies, and
output a beautifully formatted comparison table with actionable insights.
"""

PLANNING_PROMPT_INITIAL_PLAN = """
Step-by-step plan:
1. Validate that the company name exists using LLM knowledge.
2. Determine the sector using LLM knowledge.
3. Identify top 3 competitors using LLM knowledge.
4. Gather data on strategies using web search, page browsing and social media websites.
5. Analyze strategies and generate a comparison table.
6. Propose actionable insights.
Do not repeat steps unless the output becomes inaccurate or inadmissible.
"""
MANAGED_AGENT_TASK = """
Your task is to analyze the strategies of the top 3 competitors for {task_description} and
produce a comparison table with actionable insights.
"""

#Test Gemini Formatting
assembly = PromptAssembly(prompts=[
    SystemPrompt(text=SYSTEM_PROMPT),
    UserPrompt(text=PLANNING_PROMPT_INITIAL_PLAN)
])

gemini_msgs = assembly.build(provider=LLMProvider.GEMINI)
print(f"Gemini Message Output = {gemini_msgs}")

assembly = PromptAssembly(prompts=[
    SystemPrompt(text=SYSTEM_PROMPT),
    UserPrompt(text=PLANNING_PROMPT_INITIAL_PLAN)
])

openai_msgs = assembly.build(provider=LLMProvider.OPENAI)
print(f"OpenAI Message Output = {openai_msgs}")
