<!--
description: Run detached Claude Agent SDK jobs in Python with Temporal and reconnect for status, results, or cancellation.
tags: [agents, python, claude]
priority: 500
-->
# Run detached Claude Agent SDK jobs

This recipe starts a Claude Agent SDK job as a Temporal Workflow and returns its Workflow Id immediately. A later process can inspect the Workflow, wait for its result, or cancel it.

The Activity gives the agent one in-process MCP tool backed by a SQLite uniqueness constraint. If a Worker stops after the effect commits, Temporal retries the Activity without applying the effect twice.

```bash
uv sync
temporal server start-dev --db-filename temporal.db
uv run worker.py

uv run start_workflow.py submit "Record the effect, then answer done." --effect hello
uv run start_workflow.py status <workflow-id>
uv run start_workflow.py result <workflow-id>
uv run start_workflow.py cancel <workflow-id>
```

The Worker must have valid Claude Agent SDK authentication. Use `uv run worker.py --demo` to exercise the Temporal lifecycle without Claude credentials. Stop it after `effect_inserted=True`, restart it, then attach with `uv run start_workflow.py result <workflow-id>`. The retried Activity prints `effect_inserted=False`.

An Activity retry starts the Claude turn again from the original prompt. It does not resume the interrupted SDK subprocess. The stable Workflow Id prevents the demonstrated external effect from running twice.

SQLite is suitable for this local recipe. A deployed service needs a shared durable store for idempotency records.
