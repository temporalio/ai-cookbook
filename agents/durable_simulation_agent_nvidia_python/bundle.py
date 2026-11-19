"""Collect a run's four records into one directory: the Temporal Event History, the OpenShell gateway logs, the NeMo Relay
trajectories, and the OpenUSD layers, with the renders, lane checks, manifest and Workflow state next to them. The
Workflow Id, the layer stack hash and the Relay root id join them.

    uv run bundle.py <workflow-id>        ->  runs/<workflow-id>/bundle/
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys

from common import connect
from paths import run_dir
from sceneops.hashing import layer_stack_hash
from workflows.simulation_agent_workflow import SimulationAgentWorkflow


async def main() -> None:
    workflow_id = sys.argv[1]
    client = await connect()
    handle = client.get_workflow_handle_for(SimulationAgentWorkflow.run, workflow_id)
    run = run_dir(workflow_id)
    out = run / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    history = await handle.fetch_history()
    (out / "history.json").write_text(history.to_json())
    state = await handle.query(SimulationAgentWorkflow.get_state)
    (out / "state.json").write_text(state.model_dump_json(indent=2))
    for sub in ("logs", "relay", "layers", "renders", "clips", "lane"):
        if (run / sub).exists():
            shutil.copytree(run / sub, out / sub, dirs_exist_ok=True)
    if (run / "manifest.json").exists():
        shutil.copyfile(run / "manifest.json", out / "manifest.json")
    root = None
    if (run / "relay" / "propagation.json").exists():
        root = json.loads((run / "relay" / "propagation.json").read_text()).get("root_uuid")
    print(f"bundle: {out}")
    print(f"  workflow id       {workflow_id}  ({len(history.events)} history events)")
    print(f"  sandbox gens      {state.gen}, {len(state.grants)} grant(s)")
    print(f"  layer stack hash  {layer_stack_hash(state.layer_pairs())[:16]}  ({len(state.layers)} layers)")
    print(f"  relay root id     {root}")
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.parent != out:
            print(f"  {p.relative_to(out)}")


if __name__ == "__main__":
    asyncio.run(main())
