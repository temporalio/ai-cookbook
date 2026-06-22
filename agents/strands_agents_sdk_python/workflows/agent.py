from datetime import timedelta
from temporalio import workflow

from models.requests import AgentRequest, WeatherRequest

MAX_ITERATIONS = 10


@workflow.defn
class StrandsAgentWorkflow:
    @workflow.run
    async def run(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]
        iterations = 0

        # Agentic loop
        while True:
            iterations += 1
            if iterations > MAX_ITERATIONS:
                return "Agent exceeded maximum iterations"

            response = await workflow.execute_activity(
                "agent_activity",
                AgentRequest(messages=messages),
                start_to_close_timeout=timedelta(seconds=30)
            )

            if response.get("tool_calls"):
                tool_results = []
                for tool_call in response["tool_calls"]:
                    result = await self._execute_tool(tool_call["tool_name"], tool_call.get("parameters", {}))
                    tool_results.append(f"{tool_call['tool_name']}: {result}")

                messages.append({
                    "role": "assistant",
                    "content": f"Tool results: {' | '.join(tool_results)}"
                })
                continue

            if response.get("final_answer"):
                return response["final_answer"]

            return "Agent failed to provide a response"

    async def _execute_tool(self, tool_name: str, parameters: dict) -> str:
        if tool_name == "get_time":
            return await workflow.execute_activity("get_time_activity", start_to_close_timeout=timedelta(seconds=10))
        elif tool_name == "get_weather":
            return await workflow.execute_activity("get_weather_activity", WeatherRequest(**parameters), start_to_close_timeout=timedelta(seconds=10))
        elif tool_name == "list_files":
            return await workflow.execute_activity("list_files_activity", start_to_close_timeout=timedelta(seconds=10))
        return f"Unknown tool: {tool_name}"