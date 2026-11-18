"""The renderer's job registry: attach by key, progress, cancellation, a disk cache of finished jobs.

Pure Python, no Kit imports, so it is tested on a laptop. One job runs at a time: one GPU.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"


@dataclass
class Job:
    key: str
    op: str
    params: dict[str, Any]
    status: str = QUEUED
    progress: dict[str, Any] = field(default_factory=lambda: {"phase": "queued", "frames_done": 0, "frames_total": 0, "pct": 0})
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    adopters: int = 0
    cancel_requested: bool = False

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("cancel_requested")
        return d


class CancelledJob(Exception):
    pass


class Backend(Protocol):
    def run(self, job: Job, progress: Callable[[dict[str, Any]], None], cancelled: Callable[[], bool]) -> dict[str, Any]: ...


class JobRegistry:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted(cache_dir.glob("*.json")):
                try:
                    d = json.loads(p.read_text())
                    if d.get("status") == DONE:
                        self._jobs[d["key"]] = Job(**{k: d[k] for k in Job.__dataclass_fields__ if k in d})
                except Exception:  # noqa: BLE001
                    p.unlink(missing_ok=True)

    # ---- submission / attach ----
    def submit(self, key: str, op: str, params: dict[str, Any]) -> tuple[Job, bool]:
        """Returns (job, created). If the key exists in any state, attach (created=False)."""
        with self._lock:
            job = self._jobs.get(key)
            if job is not None:
                job.adopters += 1
                return job, False
            job = Job(key=key, op=op, params=params)
            self._jobs[key] = job
            return job, True

    def get(self, key: str) -> Job | None:
        with self._lock:
            return self._jobs.get(key)

    def cancel(self, key: str) -> str:
        """'cancelling' | 'done' (no-op) | 'missing'."""
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                return "missing"
            if job.status in (DONE, FAILED, CANCELLED):
                return "done"
            job.cancel_requested = True
            if job.status == QUEUED:
                job.status = CANCELLED
                job.finished_at = time.time()
                job.error = {"type": "Cancelled", "msg": "cancelled before start", "retryable": False}
            return "cancelling"

    def queue_depth(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == QUEUED)

    # ---- execution (called by the single runner) ----
    def next_queued(self) -> Job | None:
        with self._lock:
            for job in self._jobs.values():
                if job.status == QUEUED:
                    job.status = RUNNING
                    job.started_at = time.time()
                    job.progress = {**job.progress, "phase": "starting"}
                    return job
            return None

    def run_one(self, backend: Backend) -> bool:
        job = self.next_queued()
        if job is None:
            return False

        def progress(p: dict[str, Any]) -> None:
            with self._lock:
                job.progress = {**job.progress, **p}

        try:
            result = backend.run(job, progress, lambda: job.cancel_requested)
            with self._lock:
                job.status, job.result, job.finished_at = DONE, result, time.time()
                job.progress = {**job.progress, "phase": "done", "pct": 100}
            self._persist(job)
        except CancelledJob:
            with self._lock:
                job.status, job.finished_at = CANCELLED, time.time()
                job.error = {"type": "Cancelled", "msg": "cancelled while running", "retryable": False}
        except Exception as e:  # noqa: BLE001
            with self._lock:
                job.status, job.finished_at = FAILED, time.time()
                job.error = {"type": getattr(e, "error_type", type(e).__name__), "msg": str(e), "retryable": getattr(e, "retryable", True)}
        return True

    def _persist(self, job: Job) -> None:
        if self._cache_dir:
            (self._cache_dir / f"{job.key}.json").write_text(json.dumps(job.public(), sort_keys=True))

    def forget_failed(self, key: str) -> None:
        """Allow a resubmit after a failed/cancelled run (the activity retries with the same key)."""
        with self._lock:
            job = self._jobs.get(key)
            if job and job.status in (FAILED, CANCELLED):
                del self._jobs[key]
