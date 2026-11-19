"""Where the run's records live. Everything a run produces is a file under runs/<workflow-id>/ on the Worker host that ran
the Activity; Temporal payloads carry paths and hashes, never bytes."""

from __future__ import annotations

import os
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent
SITE_DIR = RECIPE_DIR / "site"
SITE_LAYER = SITE_DIR / "000_site.usda"  # the trusted copy: docks, lanes, keep-clear volumes, cameras
CATALOG = SITE_DIR / "catalog.json"  # the trusted copy: placeable Isaac Sim assets with measured footprints
ASSET_ROOT = os.environ.get("ASSET_ROOT", "/opt/assets")  # where the renderer host keeps the Isaac Sim assets; layers reference paths under it


def runs_dir() -> Path:
    return Path(os.environ.get("RUNS_DIR", "runs")).resolve()


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id
