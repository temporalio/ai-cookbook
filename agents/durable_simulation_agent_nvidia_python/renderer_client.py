"""Async client for the renderer service's localhost job API (renderer/server.py)."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable

import httpx


class RendererError(Exception):
    def __init__(self, msg: str, *, retryable: bool = True, error: dict[str, Any] | None = None) -> None:
        super().__init__(msg)
        self.retryable = retryable
        self.error = error or {}


class RendererClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or os.environ.get("RENDERER_URL", "http://127.0.0.1:8799")).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthz(self) -> dict[str, Any]:
        r = await self._client.get("/healthz")
        r.raise_for_status()
        return r.json()

    async def submit(self, key: str, op: str, params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Returns (job, attached). 409 = attached to an existing job with this key."""
        r = await self._client.post("/jobs", json={"key": key, "op": op, "params": params})
        if r.status_code in (202, 409):
            return r.json(), r.status_code == 409
        raise RendererError(f"submit {op} failed: {r.status_code} {r.text[:200]}", retryable=r.status_code >= 500)

    async def get(self, key: str) -> dict[str, Any] | None:
        r = await self._client.get(f"/jobs/{key}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    async def cancel(self, key: str) -> None:
        await self._client.delete(f"/jobs/{key}")

    async def run(self, key: str, op: str, params: dict[str, Any], *, on_progress: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
                  poll: float = 1.0, timeout: float = 1800.0, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Submit (or attach) and wait for completion. A 404 mid-poll means the service restarted → resubmit once."""
        job, attached = await self.submit(key, op, params)
        if on_progress:
            res = on_progress({"phase": "attached" if attached else "submitted", "adopters": job.get("adopters", 0)})
            if asyncio.iscoroutine(res):
                await res
        deadline = asyncio.get_running_loop().time() + timeout
        resubmitted = False
        while True:
            if cancelled and cancelled():
                await self.cancel(key)
                raise RendererError("cancelled", retryable=False)
            job = await self.get(key)
            if job is None:
                if resubmitted:
                    raise RendererError("job vanished twice (renderer restarting?)")
                resubmitted = True
                job, _ = await self.submit(key, op, params)
            status = job["status"]
            if on_progress:
                res = on_progress({**job.get("progress", {}), "status": status})
                if asyncio.iscoroutine(res):
                    await res
            if status == "done":
                return job["result"]
            if status in ("failed", "cancelled"):
                err = job.get("error") or {}
                raise RendererError(f"{op} {status}: {err.get('type')}: {err.get('msg')}", retryable=bool(err.get("retryable", True)), error=err)
            if asyncio.get_running_loop().time() > deadline:
                await self.cancel(key)
                raise RendererError(f"{op} timed out after {timeout}s")
            await asyncio.sleep(poll)
