import asyncio
import sys
from datetime import datetime, timezone

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from models.approval import ApprovalDecision


async def main():
    if len(sys.argv) < 4:
        print("Usage: python -m send_approval <workflow_id> <request_id> <approve|reject> [notes]")
        print("\nExample:")
        example = "  python -m send_approval human-in-the-loop-123 abc-def-ghi approve 'Looks good'"
        print(example)
        sys.exit(1)

    workflow_id = sys.argv[1]
    request_id = sys.argv[2]
    decision = sys.argv[3].lower()
    notes = sys.argv[4] if len(sys.argv) > 4 else None

    if decision not in ["approve", "reject"]:
        print("Decision must be 'approve' or 'reject'")
        sys.exit(1)

    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    approval_decision = ApprovalDecision(
        request_id=request_id,
        approved=(decision == "approve"),
        reviewer_notes=notes,
        decided_at=datetime.now(timezone.utc).isoformat(),
    )

    # Get workflow handle and send signal
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal("approval_decision", approval_decision)

    decision_type = 'approval' if approval_decision.approved else 'rejection'
    print(f"Sent {decision_type} signal to workflow {workflow_id}")


if __name__ == "__main__":
    asyncio.run(main())
