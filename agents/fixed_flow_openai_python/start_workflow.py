import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from models.flow import FixedFlowInput, HumanReviewInput, MortgageApplication
from worker import TASK_QUEUE
from workflows.fixed_flow_workflow import FixedFlowWorkflow


async def main() -> None:
    client = await Client.connect(
        "localhost:7233", data_converter=pydantic_data_converter
    )
    handle = await client.start_workflow(
        FixedFlowWorkflow.run,
        FixedFlowInput(
            application=MortgageApplication(
                case_id="mortgage-demo",
                credit_score=610,
                monthly_income=8_000,
                monthly_debt=3_200,
                loan_amount=360_000,
                property_value=450_000,
            )
        ),
        id="fixed-flow-mortgage-demo",
        task_queue=TASK_QUEUE,
    )

    packet = None
    while packet is None:
        packet = await handle.query(FixedFlowWorkflow.get_review_packet)
        if packet is None:
            await asyncio.sleep(0.25)

    print("Review packet:\n", packet.model_dump_json(indent=2))
    choice = input("Approve this case? [y/N] ").strip().lower()
    await handle.signal(
        FixedFlowWorkflow.submit_human_review,
        HumanReviewInput(
            reviewer="demo-reviewer",
            decision="APPROVED" if choice == "y" else "REJECTED",
            notes="Reviewed from the example starter.",
        ),
    )
    result = await handle.result()
    print("Final result:\n", result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
