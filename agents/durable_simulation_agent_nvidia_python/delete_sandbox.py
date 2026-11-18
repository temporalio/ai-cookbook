"""Chaos: delete one of the run's sandboxes out from under the agent. The next tool call sees SandboxLost, builds the next
generation with every recorded grant and every recorded layer, and continues.

    uv run delete_sandbox.py <workflow-id> [exec|tools]
"""

from __future__ import annotations

import asyncio
import sys

from activities.openshell import backend
from common import connect
from workflows.simulation_agent_workflow import SimulationAgentWorkflow


async def main() -> None:
    workflow_id = sys.argv[1]
    kind = sys.argv[2] if len(sys.argv) > 2 else "exec"
    client = await connect()
    state = await client.get_workflow_handle_for(SimulationAgentWorkflow.run, workflow_id).query(SimulationAgentWorkflow.get_state)
    name = state.exec_sandbox if kind == "exec" else state.tools_sandbox
    if not name:
        raise SystemExit(f"the run holds no {kind} sandbox right now (generation {state.gen}). Sandboxes exist only while the agent works; "
                         "they are released during a review, so run this again during the next round.")
    await backend().delete(name)
    print(f"deleted {name} (generation {state.gen}); the agent's next tool call will rebuild generation {state.gen + 1} from {len(state.layers)} recorded layer(s)")


if __name__ == "__main__":
    asyncio.run(main())
