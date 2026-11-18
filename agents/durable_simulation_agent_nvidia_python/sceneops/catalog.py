"""The asset catalog (site/catalog.json): verified Isaac Sim props with measured footprints. Only catalog ids are placeable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

S3_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"


def load_catalog(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return {a["id"]: a for a in data["assets"]}


def catalog_search(catalog: dict[str, dict[str, Any]], query: str, k: int = 8) -> list[dict[str, Any]]:
    """Keyword match over id, name and tags; the fallback when USD Search is denied or unreachable."""
    q = {w for w in query.lower().replace(",", " ").split() if w}
    scored = []
    for a in catalog.values():
        text = " ".join([a["id"], a["name"], *a["tags"]]).lower()
        score = sum(1 for w in q if w in text)
        if score:
            scored.append((score, a))
    scored.sort(key=lambda s: (-s[0], s[1]["id"]))
    return [{"id": a["id"], "name": a["name"], "size_m": a["size_m"], "tags": a["tags"], "usd": a["usd"], "source": "catalog"} for _, a in scored[:k]]


def merge_hits(hits: list[dict[str, Any]], catalog: dict[str, dict[str, Any]], query: str, k: int = 8) -> list[dict[str, Any]]:
    """USD Search hits mapped to catalog ids (by file name, then by folder), then catalog matches. Hits that map to no
    catalog entry are kept with id null: evidence that the semantic search ran, but not placeable."""
    by_stem = {a["usd"].rsplit("/", 1)[-1].rsplit(".", 1)[0].lower(): a["id"] for a in catalog.values()}
    by_folder: dict[str, str] = {}
    for a in catalog.values():
        by_folder.setdefault(a["usd"].rsplit("/", 2)[-2].lower(), a["id"])
    out, seen = [], set()
    for h in hits:
        url = str(h.get("url", ""))
        rel = url[len(S3_ROOT):] if url.startswith(S3_ROOT) else url
        stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        folder = rel.rsplit("/", 2)[-2].lower() if rel.count("/") >= 2 else ""
        cid = by_stem.get(stem) or by_folder.get(folder)
        item: dict[str, Any] = {"id": cid, "score": h.get("score"), "usd": rel, "source": "usd_search"}
        if cid:
            if cid in seen:
                continue
            seen.add(cid)
            item.update({"name": catalog[cid]["name"], "size_m": catalog[cid]["size_m"], "tags": catalog[cid]["tags"]})
        out.append(item)
    for a in catalog_search(catalog, query, k):
        if a["id"] not in seen:
            seen.add(a["id"])
            out.append(a)
    return out[: max(k, 8)]
