"""Static site facts for Workflow code, mirrored from site/000_site.usda (named site_facts: `site` is a stdlib module). Workflow code reads no files and imports no pxr;
tests/test_sceneops.py asserts these match the layer, so the sandbox, the judge and the agent's view agree."""

from __future__ import annotations

DOCKS: dict[str, list[float]] = {"1": [4.9, -16.0], "2": [4.9, -10.0], "3": [4.9, -4.0], "4": [4.9, 2.0]}

# aabb = [xmin, ymin, xmax, ymax] in metres (Z-up stage, metersPerUnit 1)
SITE_VOLUMES: list[dict] = [
    {"name": "L1", "kind": "lane", "aabb": [-26.0, 6.0, 5.4, 8.0]},
    {"name": "L2", "kind": "lane", "aabb": [2.4, -22.0, 4.4, 8.0]},
    {"name": "L3", "kind": "lane", "aabb": [-26.0, -13.0, 5.4, -11.0]},
    {"name": "KC_Dock4", "kind": "keep_clear", "aabb": [0.9, 0.0, 5.4, 4.0]},
]
LANES = ["L1", "L2", "L3"]
BUILD_AREA: list[float] = [-5.0, -5.0, 0.3, 5.4]  # /World/Site/Candidates/BuildArea
BUILD_AREA_HINT = ("open floor south of the racks between the dock lane L2 and the centre pillars; everything must fit inside "
                   "build_area_aabb (within 10 m of Dock 4 at (4.9, 2))")
BUILDING: list[float] = [-26.3, -23.4, 5.46, 30.6]  # interior of the Isaac Sim full_warehouse stage
CAMERAS = ["Station", "Wide", "TopDown", "LaneL2", "RobotEye"]
MIN_CLEARANCE_M = 0.6
