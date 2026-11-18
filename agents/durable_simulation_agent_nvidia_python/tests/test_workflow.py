"""End to end with a scripted model and the fake sandbox backend: the denied search, the operator's grant, the retried
search, a layer with a lane violation, the lane check that fails, the plan-view render, a review that asks for a
revision, the next generation rebuilt from the recorded layer with the same layer stack hash, the fix, approval, and
the four records on disk. Then: a sandbox deleted mid-round, a cancellation, and a retried apply."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta

import pytest
from temporalio.client import (
    WorkflowFailureError,
    WorkflowHandle,
    WorkflowUpdateFailedError,
)
from temporalio.exceptions import CancelledError
from temporalio.openai_agents import ModelActivityParameters
from temporalio.openai_agents.testing import AgentEnvironment
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities import gpu
from activities import sandbox as sb
from activities.openshell import FakeSandboxes
from models import AgentInput, GrantDecision, ReviewDecision
from sceneops.hashing import layer_stack_hash, sha256_file
from tests.conftest import scripted_model
from workflows.simulation_agent_workflow import SimulationAgentWorkflow
from workflows.tools import USD_SEARCH_HOST

CODE_1 = ("s1 = h.place_asset('packing_table', -3.0, 2.0, 90, name='station_1')\n"
          "s2 = h.place_asset('packing_table', -3.0, -1.0, 90, name='station_2')\n"
          "p = h.place_asset('pallet', 1.7, -3.0, 0, name='staging_1')\n"
          "h.add_light('task_light', -3.0, 0.5)\n"
          "print(h.check_clearance())\n")
CODE_2 = "moves = h.fix_all()\nprint(moves)\n"
ROUND_1 = [("inspect_site", {}), ("search_assets", {"query": "packing table pallet conveyor"}), ("apply_usd_code", {"code": CODE_1, "rationale": "place the cell"}),
           ("check_lane", {"lane": "L2"}), ("render_preview", {"camera": "Station"}), ("request_review", {"summary": "Cell placed; the pallet is close to L2."})]
ROUND_2 = [("inspect_site", {}), ("apply_usd_code", {"code": CODE_2, "rationale": "clear the lane"}), ("check_lane", {"lane": "L2"}),
           ("render_preview", {"camera": "TopDown"}), ("request_review", {"summary": "Lane clear."})]
ACTIVITIES = [sb.ensure_sandboxes, sb.release_sandboxes, sb.sandbox_search, sb.sandbox_apply, sb.apply_grant, sb.write_manifest,
              gpu.render_preview, gpu.simulate_lane, gpu.close_trace]


async def start(client, tq: str, hours: float = 72) -> WorkflowHandle:
    return await client.start_workflow(SimulationAgentWorkflow.run, AgentInput(brief="Lay out packing cell 2 near Dock 4.", model="test-model", review_timeout_hours=hours,
                                                                              gpu_task_queue=tq), id=f"packing-{uuid.uuid4().hex[:8]}", task_queue=tq,
                                       execution_timeout=timedelta(minutes=5))


async def until_review(handle: WorkflowHandle, granted: list[str]) -> str:
    """Play the operator: approve the first grant request for each host; return the pending review id."""
    for _ in range(600):
        pg = await handle.query(SimulationAgentWorkflow.pending_grant)
        if pg and pg["host"] not in granted:
            with pytest.raises(WorkflowUpdateFailedError):  # the validator rejects a decision for the wrong host
                await handle.execute_update(SimulationAgentWorkflow.grant, GrantDecision(host="evil.example", decision="approved", approver="t", notes=""))
            await handle.execute_update(SimulationAgentWorkflow.grant, GrantDecision(host=pg["host"], decision="approved", approver="test-operator", notes=""))
            granted.append(pg["host"])
        rid = await handle.query(SimulationAgentWorkflow.pending_review)
        if rid:
            return rid
        await asyncio.sleep(0.2)
    raise AssertionError("no review requested")


async def completed_results(handle: WorkflowHandle, activity_type: str) -> list[dict]:
    """The recorded results of every completed Activity of a type, straight from the Event History."""
    hist = await handle.fetch_history()
    sched = {e.event_id: e.activity_task_scheduled_event_attributes.activity_type.name for e in hist.events if e.HasField("activity_task_scheduled_event_attributes")}
    out = []
    for e in hist.events:
        if e.HasField("activity_task_completed_event_attributes"):
            a = e.activity_task_completed_event_attributes
            if sched[a.scheduled_event_id] == activity_type:
                out.append(json.loads(a.result.payloads[0].data))
    return out


async def tool_outputs(handle: WorkflowHandle, tool: str) -> list[str]:
    """What a tool returned to the model, in call order, read from the model Activities' inputs in the Event History."""
    hist = await handle.fetch_history()
    calls: dict[str, bool] = {}
    outputs: dict[str, str] = {}
    for e in hist.events:
        if not e.HasField("activity_task_scheduled_event_attributes"):
            continue
        a = e.activity_task_scheduled_event_attributes
        if a.activity_type.name != "invoke_model_activity":
            continue
        for item in json.loads(a.input.payloads[0].data).get("input") or []:
            if isinstance(item, dict) and item.get("type") == "function_call" and item.get("name") == tool:
                calls[item["call_id"]] = True
            if isinstance(item, dict) and item.get("type") == "function_call_output" and item.get("call_id") in calls:
                outputs[item["call_id"]] = item["output"]
    return [outputs[c] for c in calls if c in outputs]


@pytest.mark.timeout(120)
async def test_grant_violation_revise_rebuild_fix_approve_and_the_records(fake_env):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with AgentEnvironment(model=scripted_model(ROUND_1 + ROUND_2), model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30))) as agent_env:
            client = agent_env.applied_on_client(env.client)
            tq = f"test-{uuid.uuid4().hex[:6]}"
            async with Worker(client, task_queue=tq, workflows=[SimulationAgentWorkflow], activities=ACTIVITIES):
                handle = await start(client, tq)
                granted: list[str] = []
                rid1 = await until_review(handle, granted)

                st = await handle.query(SimulationAgentWorkflow.get_state)
                assert granted == [USD_SEARCH_HOST]
                assert [(g.host, g.decision, g.applied) for g in st.grants] == [(USD_SEARCH_HOST, "approved", True)]
                assert st.gen == 1 and st.step == 1 and [layer.step for layer in st.layers] == [1]
                assert st.lane_checks[-1].passed is False and st.lane_checks[-1].contacts == ["/World/Agent/staging_1"] and st.lane_checks[-1].method == "geometry"
                assert st.renders[-1].camera == "Station" and st.renders[-1].mode == "plan" and (fake_env / handle.id / "renders" / "001-Station-plan.png").exists()
                assert st.exec_sandbox is None and st.tools_sandbox is None, "sandboxes are released while a person decides"
                logs = fake_env / handle.id / "logs"
                assert sorted(p.name for p in logs.iterdir()) == ["g1-exec.log", "g1-tools.log"]
                tools_log = (logs / "g1-tools.log").read_text()
                assert tools_log.index("DENIED") < tools_log.index("Policy reloaded") < tools_log.index("ALLOWED")
                layer1 = fake_env / handle.id / "layers" / "001.usda"
                assert sha256_file(layer1) == st.layers[0].sha256

                with pytest.raises(WorkflowUpdateFailedError):  # a stale review id is rejected before it reaches the history
                    await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id="stale", action="approve", notes=""))
                assert await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=rid1, action="revise", notes="move the pallet out of lane L2")) == "revise"

                rid2 = await until_review(handle, granted)
                assert rid2 != rid1 and granted == [USD_SEARCH_HOST], "the agent did not ask again: the grant is state"
                st2 = await handle.query(SimulationAgentWorkflow.get_state)
                assert st2.gen == 2 and st2.step == 2 and [layer.step for layer in st2.layers] == [1, 2]
                assert st2.lane_checks[-1].passed is True and st2.renders[-1].camera == "TopDown"
                # generation 2 was built with the recorded grant and the recorded layer; the recomposed stack hashes the same
                assert "1 granted host(s): search.simready.omniverse.nvidia.com" in (logs / "g2-tools.log").read_text()
                ensures = await completed_results(handle, "ensure_sandboxes")
                assert [(e["gen"], e["rehydrated_steps"]) for e in ensures] == [(1, []), (2, [1])]
                assert ensures[1]["layer_stack_hash"] == layer_stack_hash([(1, st.layers[0].sha256)])

                # inspect_site at the start of round 2 shows round 1's layout, not an empty scene
                sites = [json.loads(o) for o in await tool_outputs(handle, "inspect_site")]
                assert [len(s["placed_assets"]) for s in sites] == [0, 3] and sites[1]["layers_applied"] == [1]

                assert await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=rid2, action="approve", notes="")) == "approve"
                result = await handle.result()
    assert result.status == "approved" and result.rounds == 2 and result.gen == 2
    assert [d["action"] for d in result.decisions] == ["revise", "approve"] and result.decisions[0]["notes"] == "move the pallet out of lane L2"
    run = fake_env / handle.id
    pairs = [(1, sha256_file(run / "layers" / "001.usda")), (2, sha256_file(run / "layers" / "002.usda"))]
    assert result.layer_stack_hash == layer_stack_hash(pairs)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["layer_stack_hash"] == result.layer_stack_hash and manifest["status"] == "approved" and len(manifest["lane_checks"]) == 2
    assert manifest["openshell"]["sandbox_generations"] == 2 and len(manifest["openshell"]["logs"]) == 4
    # the four records: history (Temporal), gateway logs (OpenShell), trajectories (Relay), layers (OpenUSD)
    relay_dir = run / "relay"
    assert (relay_dir / "propagation.json").exists() and [p.name for p in relay_dir.glob("atif-*.json")] and [p.name for p in relay_dir.glob("events-*.jsonl")]
    atifs = [json.loads(p.read_text()) for p in relay_dir.glob("atif-*.json")]
    assert len(atifs) == 1, "one process played both Workers: one scope, one trajectory"
    called = {c["function_name"] for a in atifs for s in a["steps"] for c in (s.get("tool_calls") or [])}
    called |= {r["extra"]["event_name"] for a in atifs for s in a["steps"] for r in (s.get("observation") or {}).get("results", []) if "extra" in r}
    assert {"ensure_sandboxes", "sandbox_search", "apply_grant", "sandbox_apply", "simulate_lane", "render_preview", "release_sandboxes"} <= called
    events = [json.loads(line) for line in (sorted(relay_dir.glob("events-*.jsonl"))[-1]).read_text().splitlines()]
    assert "run_closed" in {e.get("name") for e in events}
    assert sorted(p.name for p in (run / "lane").iterdir()) == ["001-L2.json", "002-L2.json"]
    assert not any(p.is_dir() for p in (fake_env / "_fake_sandboxes").iterdir()), "no sandbox outlives the run"


@pytest.mark.timeout(120)
async def test_a_deleted_sandbox_is_rebuilt_with_the_grant_mid_round(fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_LOSE_ONCE", "exec")  # the fake deletes the exec sandbox under the first apply
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with AgentEnvironment(model=scripted_model(ROUND_1), model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30))) as agent_env:
            client = agent_env.applied_on_client(env.client)
            tq = f"test-{uuid.uuid4().hex[:6]}"
            async with Worker(client, task_queue=tq, workflows=[SimulationAgentWorkflow], activities=ACTIVITIES):
                handle = await start(client, tq, hours=1)
                granted: list[str] = []
                rid = await until_review(handle, granted)
                st = await handle.query(SimulationAgentWorkflow.get_state)
                assert st.gen == 2, "SandboxLost mid-round: the tool rebuilt the next generation and retried"
                assert [layer.step for layer in st.layers] == [1] and len(st.grants) == 1
                logs = fake_env / handle.id / "logs"
                assert "1 granted host(s): search.simready.omniverse.nvidia.com" in (logs / "g2-tools.log").read_text()
                assert (logs / "g1-tools.log").exists(), "the retired generation's gateway log was kept"
                assert await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=rid, action="approve", notes="")) == "approve"
                result = await handle.result()
    assert result.status == "approved" and result.gen == 2 and result.rounds == 1


@pytest.mark.timeout(120)
async def test_cancelling_the_workflow_during_the_grant_wait_still_releases_the_sandboxes(fake_env):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with AgentEnvironment(model=scripted_model(ROUND_1), model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30))) as agent_env:
            client = agent_env.applied_on_client(env.client)
            tq = f"test-{uuid.uuid4().hex[:6]}"
            async with Worker(client, task_queue=tq, workflows=[SimulationAgentWorkflow], activities=ACTIVITIES):
                handle = await start(client, tq, hours=1)
                for _ in range(600):
                    if await handle.query(SimulationAgentWorkflow.pending_grant):
                        break
                    await asyncio.sleep(0.2)
                else:
                    raise AssertionError("no grant requested")
                sandboxes = fake_env / "_fake_sandboxes"
                assert len([p for p in sandboxes.iterdir() if p.is_dir()]) == 2, "both sandboxes exist while the agent waits for the operator"
                await handle.cancel()
                with pytest.raises(WorkflowFailureError) as info:
                    await handle.result()
                assert isinstance(info.value.cause, CancelledError)
    assert not any(p.is_dir() for p in sandboxes.iterdir()), "the cancelled round released its sandboxes"
    assert sorted(p.name for p in (fake_env / handle.id / "logs").iterdir()) == ["g1-exec.log", "g1-tools.log"]


@pytest.mark.timeout(120)
async def test_a_transient_sandbox_error_is_retried_and_the_apply_runs_once(fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_APPLY_FAIL_ONCE", "1")  # the fake's CLI "fails" before the first apply runs any code
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with AgentEnvironment(model=scripted_model(ROUND_1), model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30))) as agent_env:
            client = agent_env.applied_on_client(env.client)
            tq = f"test-{uuid.uuid4().hex[:6]}"
            async with Worker(client, task_queue=tq, workflows=[SimulationAgentWorkflow], activities=ACTIVITIES):
                handle = await start(client, tq, hours=1)
                rid = await until_review(handle, [])
                st = await handle.query(SimulationAgentWorkflow.get_state)
                assert [layer.step for layer in st.layers] == [1] and st.step == 1
                assert FakeSandboxes().apply_calls() == 2, "attempt 2 ran the code; attempt 1 failed before it could"
                applies = await completed_results(handle, "sandbox_apply")
                assert len(applies) == 1 and applies[0]["ok"] and applies[0]["cached"] is False
                assert await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=rid, action="approve", notes="")) == "approve"
                await handle.result()


@pytest.mark.timeout(120)
async def test_a_generation_that_cannot_be_rehydrated_leaves_no_sandbox_behind(fake_env):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with AgentEnvironment(model=scripted_model(ROUND_1 + ROUND_2), model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30))) as agent_env:
            client = agent_env.applied_on_client(env.client)
            tq = f"test-{uuid.uuid4().hex[:6]}"
            async with Worker(client, task_queue=tq, workflows=[SimulationAgentWorkflow], activities=ACTIVITIES):
                handle = await start(client, tq, hours=1)
                rid = await until_review(handle, [])
                (fake_env / handle.id / "layers" / "001.usda").unlink()  # as if round 2 ran on a Worker without this disk
                await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=rid, action="revise", notes="move the pallet 1 m"))
                with pytest.raises(WorkflowFailureError) as info:
                    await handle.result()
                assert info.value.cause.cause.type == "LayerMissing"
    assert not any(p.is_dir() for p in (fake_env / "_fake_sandboxes").iterdir()), "the failed generation left no sandbox behind"


async def run_one_round(fake_env, script: list[tuple[str, dict]]) -> tuple[WorkflowHandle, object]:
    """Start a run with a scripted model, approve the grant, wait for the review; returns the handle and the state."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with AgentEnvironment(model=scripted_model(script), model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30))) as agent_env:
            client = agent_env.applied_on_client(env.client)
            tq = f"test-{uuid.uuid4().hex[:6]}"
            async with Worker(client, task_queue=tq, workflows=[SimulationAgentWorkflow], activities=ACTIVITIES):
                handle = await start(client, tq, hours=1)
                rid = await until_review(handle, [])
                state = await handle.query(SimulationAgentWorkflow.get_state)
                outputs = {name: await tool_outputs(handle, name) for name in ("fix_all",)}
                hist = await handle.fetch_history()
                outputs["model_inputs"] = [e.activity_task_scheduled_event_attributes.input.payloads[0].data.decode() for e in hist.events
                                           if e.HasField("activity_task_scheduled_event_attributes")
                                           and e.activity_task_scheduled_event_attributes.activity_type.name == "invoke_model_activity"]
                await handle.execute_update(SimulationAgentWorkflow.review, ReviewDecision(review_id=rid, action="approve", notes=""))
                await handle.result()
    return outputs, state


@pytest.mark.timeout(120)
async def test_an_unknown_tool_name_goes_back_to_the_model(fake_env):
    outputs, state = await run_one_round(fake_env, [("fix_all", {}), *ROUND_1])
    assert outputs["fix_all"] == ["Tool 'fix_all' not found."], "the model saw the error and the run went on"
    assert [layer.step for layer in state.layers] == [1]


@pytest.mark.timeout(120)
async def test_review_needs_checks_of_the_current_layout(fake_env):
    script = [*ROUND_1[:5], ("apply_usd_code", {"code": CODE_2, "rationale": "clear the lane"}), ("request_review", {"summary": "too early"}),
              ("check_lane", {"lane": "L2"}), ("render_preview", {"camera": "Station"}), ("request_review", {"summary": "Lane clear."})]
    outputs, state = await run_one_round(fake_env, script)
    # request_review ends the agent's run; the refusal comes back in the Workflow's next prompt
    assert any("Your request_review was refused" in i and "on the current layout (step 2)" in i for i in outputs["model_inputs"]), "step 1's checks do not cover step 2"
    assert [c.step for c in state.lane_checks] == [1, 2] and [r.step for r in state.renders] == [1, 2]
