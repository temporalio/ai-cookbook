import json
import re
from temporalio import activity
from strands import Agent
from strands.models.bedrock import BedrockModel, BotocoreConfig

from models.requests import AgentRequest
from models.orchestrator import AgentResponse
from helpers.prompts import AGENT_SYSTEM_PROMPT


def extract_json(text: str) -> dict:
    """Extract JSON from text that may contain extra content."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in response")


@activity.defn
async def agent_activity(request: AgentRequest) -> AgentResponse:
    # Disable retries - Temporal handles them
    config = BotocoreConfig(retries={'max_attempts': 0})
    model = BedrockModel(model_id=request.model_id, config=config)
    agent = Agent(model=model, system_prompt=AGENT_SYSTEM_PROMPT)

    conversation = "\n\n".join([
        f"{msg['role']}: {msg['content']}" for msg in request.messages
    ])

    result = agent(conversation)
    result_text = result.content if hasattr(result, 'content') else str(result)

    try:
        return AgentResponse(**extract_json(result_text))
    except (json.JSONDecodeError, ValueError) as e:
        activity.logger.error(f"Failed to parse: {e}")
        return AgentResponse(tool_calls=[], final_answer=result_text, reasoning="Parsing failed")