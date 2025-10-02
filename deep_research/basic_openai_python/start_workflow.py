import asyncio
import sys
import uuid
import os
from datetime import datetime

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.deep_research_workflow import DeepResearchWorkflow


async def main():
    # Connect to Temporal server with matching data converter
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    # Get research query from command line or use default
    if len(sys.argv) > 1:
        research_query = " ".join(sys.argv[1:])
    else:
        research_query = "What are the latest developments in renewable energy policy and their economic impacts?"

    print(f"🔍 Starting Deep Research for: {research_query}")
    print("=" * 80)

    # Generate unique workflow ID (or use environment variable if set)
    workflow_id = os.environ.get("WORKFLOW_ID")
    if not workflow_id:
        workflow_id = f"deep-research-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    print(f"📋 Workflow ID: {workflow_id}")

    try:
        # Execute the deep research workflow
        result = await client.execute_workflow(
            DeepResearchWorkflow.run,
            research_query,
            id=workflow_id,
            task_queue="deep-research-task-queue",
        )

        print("📋 RESEARCH COMPLETED!")
        print("=" * 80)
        print(result)

    except Exception as e:
        print(f"❌ Research failed: {e}")
        print("Please check that:")
        print("1. Temporal server is running (temporal server start-dev)")
        print("2. Worker is running (uv run python -m worker)")
        print("3. OpenAI API key is configured")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
