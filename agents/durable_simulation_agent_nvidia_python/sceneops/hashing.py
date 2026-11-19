"""Content hashing shared by the workflow, the renderer and sceneops.

sceneops vendors this file verbatim (sandbox-image/sceneops/hashing.py); keep it
dependency-free and byte-identical in both places.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def layer_stack_hash(layers: list[tuple[int, str]]) -> str:
    """Hash of the ordered layer stack: [(step, sha256), ...] oldest first."""
    joined = "\n".join(f"{step:03d}:{sha}" for step, sha in layers)
    return hashlib.sha256(joined.encode()).hexdigest()


def job_key(op: str, base_ref: str, layer_shas: list[str], params: dict[str, Any]) -> str:
    """Idempotency key for a renderer job; computed in the activity so retries attach."""
    return hashlib.sha256(
        canonical_json({"op": op, "base_ref": base_ref, "layers": layer_shas, "params": params}).encode()
    ).hexdigest()
