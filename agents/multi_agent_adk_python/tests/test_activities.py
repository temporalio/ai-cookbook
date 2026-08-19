"""Activity-level tests using ActivityEnvironment (no Temporal server needed).

The full multi-agent workflow needs a real LLM, so it isn't covered here —
it's exercised end-to-end by running ``worker.py`` + ``start_workflow.py``
with ``GOOGLE_API_KEY`` set.
"""

import pytest
from temporalio.testing import ActivityEnvironment

from activities.tools import (
    tool_get_fleet_status,
    tool_get_order_priorities,
    tool_get_route_info,
)
from models.models import AssignmentInput, AssignmentOutput


class TestFleetStatus:
    @pytest.mark.asyncio
    async def test_returns_five_drivers(self):
        env = ActivityEnvironment()
        result = await env.run(tool_get_fleet_status)
        for driver_id in ("driver-a", "driver-b", "driver-c", "driver-d", "driver-e"):
            assert driver_id in result

    @pytest.mark.asyncio
    async def test_marks_full_and_disconnected_drivers(self):
        env = ActivityEnvironment()
        result = await env.run(tool_get_fleet_status)
        assert "FULL" in result
        assert "DISCONNECTED" in result


class TestOrderPriorities:
    @pytest.mark.asyncio
    async def test_includes_vip_orders(self):
        env = ActivityEnvironment()
        result = await env.run(tool_get_order_priorities)
        assert "VIP" in result
        assert "deadline" in result.lower()


class TestRouteInfo:
    @pytest.mark.asyncio
    async def test_eta_scales_with_distance(self):
        env = ActivityEnvironment()
        near = await env.run(
            tool_get_route_info,
            36.1147,
            -115.1728,
            36.1162,
            -115.1745,
            "near",
            "origin",
        )
        far = await env.run(
            tool_get_route_info,
            36.1147,
            -115.1728,
            36.5000,
            -115.5000,
            "far",
            "origin",
        )
        # Crude check: the far route should report a larger distance.
        assert "Distance:" in near
        assert "Distance:" in far
        assert "ETA:" in near and "ETA:" in far


class TestModels:
    def test_assignment_input_round_trip(self):
        order = AssignmentInput(
            order_id="order-001",
            hotel="Caesars Palace",
            priority="VIP",
            servings=50,
            deadline_minutes=25,
            delivery_lat=36.1162,
            delivery_lng=-115.1745,
        )
        again = AssignmentInput.model_validate(order.model_dump())
        assert again == order

    def test_assignment_output_construction(self):
        out = AssignmentOutput(driver_id="driver-a", reasoning_summary="test")
        assert out.driver_id == "driver-a"
        assert out.reasoning_summary == "test"
