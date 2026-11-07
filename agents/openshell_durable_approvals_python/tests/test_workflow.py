"""End to end with a scripted model and the fake sandbox backend: a denied search, the operator's grant, the retried
search, the artifact, a review that asks for a revision, the next generation built with the grant, and approval. A second
run loses its sandbox mid-round and rebuilds."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta

import pytest
from temporalio.client import WorkflowHandle, WorkflowUpdateFailedError
from temporalio.openai_agents import ModelActivityParameters
from temporalio.openai_agents.testing import (
    AgentEnvironment,
    ResponseBuilders,
    TestModel,
)
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities import sandbox as sb
from models import AgentInput, GrantDecision, ReviewDecision
from workflows.approval_agent_workflow import ApprovalAgentWorkflow
from workflows.tools import USD_SEARCH_HOST

CODE_1 = ("import json\n"
          "layout = [{'name': 'table_1', 'asset': 'packing_table', 'x_m': 0.0, 'y_m': 0.0}, {'name': 'pallet_1', 'asset': 'pallet', 'x_m': 3.0, 'y_m': 0.0}]\n"
          "json.dump(layout, open('layout.json', 'w'))\nprint('wrote', len(layout))\n")
CODE_2 = ("import json\nlayout = json.load(open('layout.json'))\nlayout[1]['x_m'] = 4.0\n"
          "json.dump(layout, open('layout.json', 'w'))\nprint('moved the pallet')\n")
ROUND_1 = [("search_assets", {"query": "packing table pallet"}), ("run_code", {"code": CODE_1, "purpose": "write layout.json"}),
           ("request_review", {"summary": "Two placements written."})]
ROUND_2 = [("run_code", {"code": CODE_2, "purpose": "move the pallet 1 m"}), ("request_review", {"summary": "Pallet moved."})]
ACTIVITIES = [sb.ensure_sandboxes, sb.release_sandboxes, sb.sandbox_search, sb.sandbox_run_code, sb.apply_grant]


def scripted_model(script: list[tuple[str, dict]]) -> TestModel:
    counter = iter(range(1, 100))

    def tool_call(name: str, args: dict):
        i = next(counter)
        return ResponseBuilders.tool_call(json.dumps(args), name, call_id=f"call_{i}", item_id=f"fc_{i}")

    return TestModel.returning_responses([tool_call(n, a) for n, a in script])


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_BACKEND", "fake")
    monkeypatch.delenv("FAKE_LOSE_ONCE", raising=False)
    return tmp_path


async def until_review(handle: WorkflowHandle, granted: list[str]) -> str:
    """Play the operator: approve the first grant request for each host; return the pending review id."""
    for _ in range(600):
        pg = await handle.query(ApprovalAgentWorkflow.pending_grant)
        if pg and pg["host"] not in granted:
            with pytest.raises(WorkflowUpdateFailedError):  # the validator rejects a decision for the wrong host
                await handle.execute_update(ApprovalAgentWorkflow.grant, GrantDecision(host="evil.example", decision="approved", approver="t", notes=""))
            await handle.execute_update(ApprovalAgentWorkflow.grant, GrantDecision(host=pg["host"], decision="approved", approver="test-operator", notes=""))
            granted.append(pg["host"])
        rid = await handle.query(ApprovalAgentWorkflow.pending_review)
        if rid:
            return rid
        await asyncio.sleep(0.2)
    raise AssertionError("no review requested")


@pytest.mark.timeout(120)
async def test_denied_search_grant_revise_rebuild_and_approve(fake_env):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with AgentEnvironment(model=scripted_model(ROUND_1 + ROUND_2), model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30))) as agent_env:
            client = agent_env.applied_on_client(env.client)
            tq = f"test-{uuid.uuid4().hex[:6]}"
            async with Worker(client, task_queue=tq, workflows=[ApprovalAgentWorkflow], activities=ACTIVITIES):
                handle = await client.start_workflow(ApprovalAgentWorkflow.run, AgentInput(brief="Lay out a packing cell.", model="test-model", review_timeout_hours=72),
                                                     id=f"packing-{uuid.uuid4().hex[:8]}", task_queue=tq, execution_timeout=timedelta(minutes=5))
                granted: list[str] = []
                rid1 = await until_review(handle, granted)

                st = await handle.query(ApprovalAgentWorkflow.get_state)
                assert granted == [USD_SEARCH_HOST]
                assert [(g.host, g.decision, g.applied) for g in st.grants] == [(USD_SEARCH_HOST, "approved", True)]
                assert [a.name for a in st.artifacts] == ["layout.json"] and st.gen == 1 and st.step == 1
                assert st.exec_sandbox is None and st.tools_sandbox is None, "sandboxes are released while a person decides"
                logs = fake_env / handle.id / "logs"
                assert sorted(p.name for p in logs.iterdir()) == ["g1-exec.log", "g1-tools.log"]
                tools_log = (logs / "g1-tools.log").read_text()
                assert tools_log.index("DENIED") < tools_log.index("Policy reloaded") < tools_log.index("ALLOWED")

                with pytest.raises(WorkflowUpdateFailedError):  # a stale review id is rejected before it reaches the history
                    await handle.execute_update(ApprovalAgentWorkflow.review, ReviewDecision(review_id="stale", action="approve", notes=""))
                assert await handle.execute_update(ApprovalAgentWorkflow.review, ReviewDecision(review_id=rid1, action="revise", notes="move the pallet 1 m")) == "revise"

                rid2 = await until_review(handle, granted)
                assert rid2 != rid1 and granted == [USD_SEARCH_HOST], "the agent did not ask again: the grant is state"
                st2 = await handle.query(ApprovalAgentWorkflow.get_state)
                assert st2.gen == 2 and st2.step == 2
                # generation 2 was built with the recorded grant, and layout.json was put back for the agent's code to read
                assert "1 granted host(s): search.simready.omniverse.nvidia.com" in (logs / "g2-tools.log").read_text()
                assert json.loads((fake_env / handle.id / "artifacts" / "layout.json").read_text())[1]["x_m"] == 4.0

                assert await handle.execute_update(ApprovalAgentWorkflow.review, ReviewDecision(review_id=rid2, action="approve", notes="")) == "approve"
                result = await handle.result()
    assert result.status == "approved" and result.rounds == 2 and result.gen == 2
    assert [d["action"] for d in result.decisions] == ["revise", "approve"] and result.decisions[0]["notes"] == "move the pallet 1 m"
    assert not (fake_env / "_fake_sandboxes").exists() or not any((fake_env / "_fake_sandboxes").iterdir()), "no sandbox outlives the run"


@pytest.mark.timeout(120)
async def test_a_deleted_sandbox_is_rebuilt_with_the_grant_mid_round(fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_LOSE_ONCE", "exec")  # the fake deletes the exec sandbox under the first run_code
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with AgentEnvironment(model=scripted_model(ROUND_1), model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30))) as agent_env:
            client = agent_env.applied_on_client(env.client)
            tq = f"test-{uuid.uuid4().hex[:6]}"
            async with Worker(client, task_queue=tq, workflows=[ApprovalAgentWorkflow], activities=ACTIVITIES):
                handle = await client.start_workflow(ApprovalAgentWorkflow.run, AgentInput(brief="Lay out a packing cell.", model="test-model", review_timeout_hours=1),
                                                     id=f"packing-{uuid.uuid4().hex[:8]}", task_queue=tq, execution_timeout=timedelta(minutes=5))
                granted: list[str] = []
                rid = await until_review(handle, granted)
                st = await handle.query(ApprovalAgentWorkflow.get_state)
                assert st.gen == 2, "SandboxLost mid-round: the tool rebuilt the next generation and retried"
                assert [a.name for a in st.artifacts] == ["layout.json"] and len(st.grants) == 1
                logs = fake_env / handle.id / "logs"
                assert "1 granted host(s): search.simready.omniverse.nvidia.com" in (logs / "g2-tools.log").read_text()
                assert (logs / "g1-tools.log").exists(), "the retired generation's gateway log was kept"
                assert await handle.execute_update(ApprovalAgentWorkflow.review, ReviewDecision(review_id=rid, action="approve", notes="")) == "approve"
                result = await handle.result()
    assert result.status == "approved" and result.gen == 2 and result.rounds == 1
