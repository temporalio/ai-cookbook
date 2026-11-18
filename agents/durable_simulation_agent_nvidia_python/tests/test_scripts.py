"""The operator scripts' plumbing: the kill script finds the Worker that runs the in-flight Activity by the identity
Temporal reports for the attempt."""

from __future__ import annotations

import json
import socket

import pytest

from kill_worker import _record_for


def test_the_kill_script_targets_the_worker_temporal_reports(tmp_path):
    records = tmp_path / "_workers"
    records.mkdir()
    for pid, queues in ((111, ["agent"]), (222, ["gpu"])):
        (records / f"{pid}.json").write_text(json.dumps({"pid": pid, "argv": queues, "queues": queues, "env": {}, "secrets": []}))
    info, record = _record_for(f"111@{socket.gethostname()}", records)
    assert info["queues"] == ["agent"] and record.name == "111.json", "the agent Worker, not whichever started last"
    with pytest.raises(SystemExit, match="not on a Worker on this host"):
        _record_for("111@some-other-host", records)
    with pytest.raises(SystemExit, match="no restart record"):
        _record_for(f"333@{socket.gethostname()}", records)
