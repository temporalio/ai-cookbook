"""Chaos: delete one of the run's sandboxes out from under the agent. The next tool call sees SandboxLost, builds the next
generation with every recorded grant, and continues.

    uv run python -m delete_sandbox <workflow-id> [exec|tools]
"""

from __future__ import annotations

import asyncio
import sys

from activities.openshell import backend
from common import connect
from workflows.approval_agent_workflow import ApprovalAgentWorkflow


async def main() -> None:
    workflow_id = sys.argv[1]
    kind = sys.argv[2] if len(sys.argv) > 2 else "exec"
    client = await connect()
    state = await client.get_workflow_handle_for(ApprovalAgentWorkflow.run, workflow_id).query(ApprovalAgentWorkflow.get_state)
    name = state.exec_sandbox if kind == "exec" else state.tools_sandbox
    if not name:
        raise SystemExit(f"the run holds no {kind} sandbox right now (generation {state.gen})")
    await backend().delete(name)
    print(f"deleted {name} (generation {state.gen}); the agent's next tool call will rebuild generation {state.gen + 1}")


if __name__ == "__main__":
    asyncio.run(main())
