"""The operator's decision on a denied egress: a Workflow Update, validated by the Workflow before it enters the history.

    uv run grant.py <workflow-id> approved|rejected [--approver name] [--notes "..."]
"""

from __future__ import annotations

import argparse
import asyncio

from common import connect
from models import GrantDecision
from workflows.simulation_agent_workflow import SimulationAgentWorkflow


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow_id")
    ap.add_argument("decision", choices=["approved", "rejected"])
    ap.add_argument("--approver", default="operator")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    client = await connect()
    handle = client.get_workflow_handle_for(SimulationAgentWorkflow.run, args.workflow_id)
    pending = await handle.query(SimulationAgentWorkflow.pending_grant)
    if not pending:
        raise SystemExit("no grant is pending")
    print(f"pending: {pending['host']}:{pending['port']} — {pending['reason']}")
    result = await handle.execute_update(SimulationAgentWorkflow.grant, GrantDecision(host=pending["host"], decision=args.decision, approver=args.approver, notes=args.notes))
    print(f"{pending['host']}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
