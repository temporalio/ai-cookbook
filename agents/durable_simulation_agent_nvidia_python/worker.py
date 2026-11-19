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

RESTART_ENV = ("SANDBOX_BACKEND", "TASK_QUEUE", "GPU_TASK_QUEUE", "TEMPORAL_ADDRESS", "TEMPORAL_NAMESPACE", "NIM_BASE_URL", "NIM_MODEL",
               "NIM_FALLBACK_MODEL", "NEMO_RELAY", "NEMO_RELAY_IDLE_CLOSE_S", "FAULT_MODEL_CALLS", "OPENSHELL_BIN", "SANDBOX_IMAGE",
               "SANDBOX_PYTHON", "RENDERER_URL", "RENDERER_SHARED_DIR", "RENDERER_BASE_REF", "ASSET_ROOT")
SECRETS = ("NVIDIA_API_KEY", "TEMPORAL_API_KEY")


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
    # For kill_worker.py: how to start this Worker again with the same settings. Secrets stay off disk; the restart takes
    # them from the shell that runs kill_worker.py.
    # One record per Worker process, keyed by pid: the pid is in the identity Temporal reports for every Activity attempt.
    env = {k: os.environ[k] for k in RESTART_ENV if k in os.environ} | {"RUNS_DIR": str(runs_dir())}
    records = runs_dir() / "_workers"
    records.mkdir(parents=True, exist_ok=True)
    (records / f"{os.getpid()}.json").write_text(json.dumps({"pid": os.getpid(), "argv": sys.argv[1:], "queues": which, "env": env,
                                                             "secrets": [k for k in SECRETS if os.environ.get(k)]}))
    await asyncio.gather(*(w.run() for w in workers))


if __name__ == "__main__":
    asyncio.run(main())
