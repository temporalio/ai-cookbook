import asyncio
import sys

from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStreamClient

from activities.tools import MODEL_EVENTS_TOPIC, TOOL_EVENTS_TOPIC


async def main():
    if len(sys.argv) != 2:
        print("Usage: uv run python -m subscribe <workflow-id>")
        sys.exit(1)
    workflow_id = sys.argv[1]

    client = await Client.connect("localhost:7233")
    stream_client = WorkflowStreamClient.create(client, workflow_id)

    print(f"Subscribing to workflow {workflow_id!r} ...")
    print(80 * "-")

    async for item in stream_client.subscribe([MODEL_EVENTS_TOPIC, TOOL_EVENTS_TOPIC]):
        if item.topic == TOOL_EVENTS_TOPIC:
            print(f"\n{item.data}")
            continue

        # item.topic == MODEL_EVENTS_TOPIC: raw OpenAI Responses API stream
        # events, decoded as plain dicts (see README for why).
        event = item.data
        event_type = event.get("type")

        if event_type == "response.output_text.delta":
            print(event.get("delta", ""), end="", flush=True)
        elif event_type == "response.output_item.done":
            item_data = event.get("item", {})
            if item_data.get("type") == "function_call":
                name = item_data.get("name")
                arguments = item_data.get("arguments")
                print(f"\n-> model requested tool call: {name}({arguments})")
        elif event_type == "response.completed":
            print()


if __name__ == "__main__":
    asyncio.run(main())
