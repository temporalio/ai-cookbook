"""The reviewer's decision on the agent's result: a Workflow Update with the pending review id, validated by the workflow.

    uv run python -m review <workflow-id> approve|revise|reject [--notes "..."]
"""

from __future__ import annotations

import argparse
import asyncio

from common import connect
from models import ReviewDecision
from workflows.approval_agent_workflow import ApprovalAgentWorkflow


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow_id")
    ap.add_argument("action", choices=["approve", "revise", "reject"])
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    client = await connect()
    handle = client.get_workflow_handle_for(ApprovalAgentWorkflow.run, args.workflow_id)
    review_id = await handle.query(ApprovalAgentWorkflow.pending_review)
    if not review_id:
        raise SystemExit("no review is pending")
    result = await handle.execute_update(ApprovalAgentWorkflow.review, ReviewDecision(review_id=review_id, action=args.action, notes=args.notes))
    print(f"{review_id}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
