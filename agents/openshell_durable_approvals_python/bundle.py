"""Collect a run's records into one directory: the Temporal event history, the sandboxes' gateway logs, the artifacts,
and the workflow's state. The run id and sandbox names join them.

    uv run python -m bundle <workflow-id>        ->  runs/<workflow-id>/bundle/
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys

from activities.openshell import runs_dir
from common import connect
from workflows.approval_agent_workflow import ApprovalAgentWorkflow


async def main() -> None:
    workflow_id = sys.argv[1]
    client = await connect()
    handle = client.get_workflow_handle_for(ApprovalAgentWorkflow.run, workflow_id)
    run = runs_dir() / workflow_id
    out = run / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    history = await handle.fetch_history()
    (out / "history.json").write_text(history.to_json())
    state = await handle.query(ApprovalAgentWorkflow.get_state)
    (out / "state.json").write_text(state.model_dump_json(indent=2))
    for sub in ("logs", "artifacts"):
        if (run / sub).exists():
            shutil.copytree(run / sub, out / sub, dirs_exist_ok=True)
    events = len(history.events)
    print(f"bundle: {out}")
    print(f"  history.json   {events} events")
    print(f"  state.json     generation {state.gen}, {len(state.grants)} grant(s), {len(state.artifacts)} artifact(s)")
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.parent != out:
            print(f"  {p.relative_to(out)}")
    print(json.dumps({"grants": [g.model_dump() for g in state.grants]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
