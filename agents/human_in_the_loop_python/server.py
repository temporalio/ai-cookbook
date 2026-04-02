import asyncio
import logging
from typing import Protocol

from aiohttp import web


class HealthProvider(Protocol):
    def is_busy(self) -> bool: ...
    def last_update_time(self) -> int: ...


async def handle_ping(request: web.Request) -> web.Response:
    provider: HealthProvider = request.app["health_provider"]
    status = "HealthyBusy" if provider.is_busy() else "Healthy"
    return web.json_response({"status": status, "time_of_last_update": provider.last_update_time()})


async def handle_invocations(request: web.Request) -> web.Response:
    return web.json_response({})


async def run(health_provider: HealthProvider):
    app = web.Application()
    app["health_provider"] = health_provider
    app.router.add_get("/ping", handle_ping)
    app.router.add_post("/invocations", handle_invocations)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    logging.getLogger(__name__).info("HTTP server listening on port 8080")

    await asyncio.Future()  # run forever
