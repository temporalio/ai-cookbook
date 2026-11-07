"""The operator's decision on a denied egress: a Workflow Update, validated by the workflow before it enters the history.

    uv run python -m grant <workflow-id> approved|rejected [--approver name] [--notes "..."]
"""

from __future__ import annotations

import argparse
import asyncio

from common import connect
from models import GrantDecision
from workflows.approval_agent_workflow import ApprovalAgentWorkflow


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow_id")
    ap.add_argument("decision", choices=["approved", "rejected"])
    ap.add_argument("--approver", default="operator")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    client = await connect()
    handle = client.get_workflow_handle_for(ApprovalAgentWorkflow.run, args.workflow_id)
    pending = await handle.query(ApprovalAgentWorkflow.pending_grant)
    if not pending:
        raise SystemExit("no grant is pending")
    print(f"pending: {pending['host']}:{pending['port']} — {pending['reason']}")
    result = await handle.execute_update(ApprovalAgentWorkflow.grant, GrantDecision(host=pending["host"], decision=args.decision, approver=args.approver, notes=args.notes))
    print(f"{pending['host']}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
