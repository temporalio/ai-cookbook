import asyncio
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.classify_workflow import ClassifyContentWorkflow
from models.signals import ContentSignals


async def main():
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # Example 1: LLM would likely say "safe", but the email address triggers a hard rule.
    print("--- Example 1: Hard rule override ---")
    signals = ContentSignals(
        text="Great product! Contact me at john.doe@example.com for a special deal.",
        author_id="user-123",
    )
    print(f"Input: {signals.text!r}")
    result = await client.execute_workflow(
        ClassifyContentWorkflow.run,
        signals,
        id="guardrails-example-1",
        task_queue="guardrails-hard-rules-task-queue",
    )
    print(f"Classification: {result.classification}")
    print(f"Overridden by hard rule: {result.overridden_by_hard_rule}")
    print(f"Reasoning: {result.reasoning}\n")

    # Example 2: Clean content — no hard rules fire, LLM verdict stands.
    print("--- Example 2: LLM verdict stands ---")
    signals = ContentSignals(
        text="I really enjoyed the hiking trail last weekend. The views were amazing!",
        author_id="user-456",
    )
    print(f"Input: {signals.text!r}")
    result = await client.execute_workflow(
        ClassifyContentWorkflow.run,
        signals,
        id="guardrails-example-2",
        task_queue="guardrails-hard-rules-task-queue",
    )
    print(f"Classification: {result.classification}")
    print(f"Overridden by hard rule: {result.overridden_by_hard_rule}")
    print(f"Reasoning: {result.reasoning}\n")


if __name__ == "__main__":
    asyncio.run(main())
