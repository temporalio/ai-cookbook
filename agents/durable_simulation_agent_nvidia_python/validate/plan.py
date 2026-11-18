"""The CPU preview: a plan view of the composed layers drawn with Pillow. The same tool call renders with Isaac Sim RTX
when a renderer is configured; without one, the reviewer still gets a picture of what the agent placed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from validate.judge import compose

VIEW = [-13.0, -14.0, 6.0, 10.0]  # x0, y0, x1, y1 in metres: the dock side of the warehouse around the build area
COLORS = {"lane": (215, 220, 228), "keep_clear": (250, 205, 205), "build": (225, 240, 225), "asset": (70, 110, 190), "floor_marking": (235, 190, 20),
          "light": (250, 220, 120), "dock": (120, 120, 120), "violation": (220, 40, 40)}


def render_plan(layer_paths: list[str | Path], *, site_layer: str | Path, catalog: str | Path, out: Path, asset_root: str = "/opt/assets",
                width: int = 1280, height: int = 720, title: str = "", violations: list[dict[str, Any]] | None = None) -> Path:
    h = compose(layer_paths, site_layer=site_layer, catalog=catalog, asset_root=asset_root)
    x0, y0, x1, y1 = VIEW
    scale = min(width / (x1 - x0), height / (y1 - y0))

    def px(x: float, y: float) -> tuple[float, float]:  # metres -> pixels, y up
        return ((x - x0) * scale, height - (y - y0) * scale)

    def rect(aabb: list[float], fill=None, outline=None, w: int = 2) -> None:
        (ax, ay), (bx, by) = px(aabb[0], aabb[3]), px(aabb[2], aabb[1])
        draw.rectangle([ax, ay, bx, by], fill=fill, outline=outline, width=w)

    img = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    for v in h.site_volumes():
        rect(v["aabb"], fill=COLORS[v["kind"]])
        draw.text(px(v["aabb"][0] + 0.2, v["aabb"][3] - 0.2), v["name"], fill=(90, 90, 90))
    if (ba := h.build_area()):
        rect(ba, outline=COLORS["build"], w=3)
    for n, (dx, dy) in h.summary()["docks"].items():
        rect([dx - 0.4, dy - 0.4, dx + 0.4, dy + 0.4], fill=COLORS["dock"])
        draw.text(px(dx - 0.3, dy + 0.9), f"Dock {n}", fill=(60, 60, 60))
    bad = {v["asset"] for v in (violations or [])}
    for prim in h.agent_prims():
        if not prim["aabb"]:
            continue
        kind = prim["kind"] if prim["kind"] in COLORS else "asset"
        rect(prim["aabb"], fill=COLORS[kind], outline=COLORS["violation"] if prim["path"] in bad else (40, 40, 40), w=3 if prim["path"] in bad else 1)
        draw.text(px(prim["aabb"][0], prim["aabb"][3] + 0.1), prim["path"].rsplit("/", 1)[-1], fill=(20, 20, 20))
    draw.text((12, 10), title or "plan view (CPU): lanes grey, keep-clear red, build area green, agent assets blue", fill=(30, 30, 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out
