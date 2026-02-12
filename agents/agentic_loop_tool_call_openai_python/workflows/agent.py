from temporalio import workflow
from datetime import timedelta

import json

with workflow.unsafe.imports_passed_through():
    from tools import get_tools
    from helpers import tool_helpers
    from activities import openai_responses

@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, input: str) -> str:

        input_list = [{"type": "message", "role": "user", "content": input}]

        # The agentic loop
        while True:

            print(80 * "=")
                
            # consult the LLM
            result = await workflow.execute_activity(
                openai_responses.create,
                openai_responses.OpenAIResponsesRequest(
                    model="gpt-4o-mini",
                    instructions=tool_helpers.HELPFUL_AGENT_SYSTEM_INSTRUCTIONS,
                    input=input_list,
                    tools=get_tools(),
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )

            # For this simple example, we only have one item in the output list
            # Either the LLM will have chosen a single function call or it will
            # have chosen to respond with a message.
            item = result.output[0]

            # Now process the LLM output to either call a tool or respond with a message.
            
            # if the result is a tool call, call the tool
            if item.type == "function_call":
                result = await self._handle_function_call(item, result, input_list)
                
                # add the tool call result to the input list for context
                input_list.append({"type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": result})

            # if the result is not a tool call we will just respond with a message
            else:
                print(f"No tools chosen, responding with a message: {result.output_text}")
                return result.output_text


    async def _handle_function_call(self, item, result, input_list):
        # serialize the LLM output - the decision the LLM made to call a tool
        i = result.output[0]
        input_list += [
            i.model_dump() if hasattr(i, "model_dump") else i
        ]
        # execute dynamic activity with the tool name chosen by the LLM
        # and the arguments crafted by the LLM
        args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments

        result = await workflow.execute_activity(
            item.name,
            args,
            start_to_close_timeout=timedelta(seconds=30),
        )

        print(f"Made a tool call to {item.name}")

        return result
 