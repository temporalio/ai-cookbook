"""Activities exposed to ADK agents as tools.

Each ``@activity.defn`` runs as a Temporal Activity — durable, retryable,
visible in the Temporal UI. The agent calls these via ``activity_tool``
wrappers (see ``_activity_tool.py``), so every tool call becomes a
durable Temporal event.

Bodies return canned data so the recipe runs without any backing service.
In a real fleet system these would query a database, hit a routing API,
or call an internal microservice.
"""

from __future__ import annotations

from temporalio import activity


@activity.defn
async def tool_get_fleet_status() -> str:
    """Return current fleet state: driver positions, capacity, and status."""
    return (
        "Fleet status:\n"
        "- driver-a: pos=(36.1147, -115.1728)  capacity=2/3  status=AVAILABLE\n"
        "- driver-b: pos=(36.1099, -115.1750)  capacity=0/3  status=AVAILABLE\n"
        "- driver-c: pos=(36.1162, -115.1745)  capacity=3/3  status=FULL\n"
        "- driver-d: pos=(36.1213, -115.1700)  capacity=1/3  status=AVAILABLE\n"
        "- driver-e: pos=(36.1080, -115.1760)  capacity=2/3  status=DISCONNECTED"
    )


@activity.defn
async def tool_get_order_priorities() -> str:
    """Return priority context: VIP vs standard, deadlines, hotel events."""
    return (
        "Open orders:\n"
        "- order-001: Caesars Palace — VIP, 50 servings, deadline 25min "
        "(gala tonight)\n"
        "- order-002: MGM Grand — standard, 12 servings, deadline 60min\n"
        "- order-003: Bellagio — VIP, 30 servings, deadline 40min "
        "(rooftop event)"
    )


@activity.defn
async def tool_get_route_info(
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float,
    destination_name: str = "",
    origin_name: str = "",
) -> str:
    """Return driving distance + ETA between two points.

    Multi-arg activity — exercises ``activity_tool``'s ``args=[...]`` path.
    A real implementation would call Google Maps / Mapbox / OSRM here.
    """
    # Crude haversine-ish estimate so the canned output varies with inputs.
    dlat = destination_lat - origin_lat
    dlng = destination_lng - origin_lng
    rough_km = ((dlat * 111) ** 2 + (dlng * 96) ** 2) ** 0.5
    eta_minutes = max(1, int(rough_km * 2))
    dest_label = destination_name or f"({destination_lat:.4f}, {destination_lng:.4f})"
    origin_label = origin_name or f"({origin_lat:.4f}, {origin_lng:.4f})"
    return (
        f"Route {origin_label} → {dest_label}:\n"
        f"  Distance: {rough_km:.2f} km\n"
        f"  ETA: {eta_minutes} min"
    )
