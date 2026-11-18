"""The Workers. `agent`: the Workflow, the sandbox Activities and the model Activity the Agents plugin registers. `gpu`:
previews and lane checks, next to the renderer (RENDERER_URL) or as the CPU stand-in.

    uv run worker.py            # both Task Queues in one process: a laptop
    uv run worker.py agent      # the agent Worker only
    uv run worker.py gpu        # the gpu Worker only, on the host that has the renderer
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from temporalio.worker import Worker

from activities import gpu
from activities import sandbox as sb
from activities.openshell import backend
from common import GPU_TASK_QUEUE, TASK_QUEUE, connect
from paths import runs_dir
from workflows.simulation_agent_workflow import SimulationAgentWorkflow


async def main() -> None:
    which = sys.argv[1:] or ["agent", "gpu"]
    sandboxes = backend() if "agent" in which else None  # fails fast with a clear message when neither OpenShell nor SANDBOX_BACKEND=fake is set
    client = await connect(register_model_activities="agent" in which)
    workers = []
    if "agent" in which:
        workers.append(Worker(client, task_queue=TASK_QUEUE, workflows=[SimulationAgentWorkflow],
                              activities=[sb.ensure_sandboxes, sb.release_sandboxes, sb.sandbox_search, sb.sandbox_apply, sb.apply_grant, sb.write_manifest]))
        print(f"agent worker on task queue {TASK_QUEUE!r}; sandbox backend: {type(sandboxes).__name__}"
              + (" (no isolation: model-written code runs on this host)" if type(sandboxes).__name__ == "FakeSandboxes" else ""))
    if "gpu" in which:
        workers.append(Worker(client, task_queue=GPU_TASK_QUEUE, activities=[gpu.render_preview, gpu.simulate_lane, gpu.close_trace]))
        print(f"gpu worker on task queue {GPU_TASK_QUEUE!r}; renderer: {gpu.renderer_url() or 'none (CPU plan views and geometry lane checks)'}")
    runs_dir().mkdir(parents=True, exist_ok=True)
    (runs_dir() / "_worker.json").write_text(json.dumps({"pid": os.getpid(), "argv": sys.argv[1:]}))  # for kill_worker.py
    await asyncio.gather(*(w.run() for w in workers))


if __name__ == "__main__":
    asyncio.run(main())
