"""The reviewer's decision on the agent's result: a Workflow Update with the pending review id, validated by the Workflow.

    uv run review.py <workflow-id> approve|revise|reject [--notes "..."]
"""

from __future__ import annotations

import argparse
import asyncio

from common import connect
from models import ReviewDecision
from workflows.simulation_agent_workflow import SimulationAgentWorkflow


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow_id")
    ap.add_argument("action", choices=["approve", "revise", "reject"])
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    client = await connect()
    handle = client.get_workflow_handle_for(SimulationAgentWorkflow.run, args.workflow_id)
    review_id = await handle.query(SimulationAgentWorkflow.pending_review)
    if not review_id:
        raise SystemExit("no review is pending")
    state = await handle.query(SimulationAgentWorkflow.get_state)
    print(f"{len(state.layers)} layer(s), {len(state.renders)} render(s) and {len(state.lane_checks)} lane check(s) in runs/{state.run_id}/")
    result = await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=review_id, action=args.action, notes=args.notes))
    print(f"{review_id}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
