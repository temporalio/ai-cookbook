"""Mirror what the renderer opens from NVIDIA's public Isaac Sim asset bucket into a local directory, so renders never
depend on the network and the agent's layers can only reference local paths (the lint refuses URLs). No credentials:
S3 ListObjectsV2 and GET.

    uv run renderer/mirror_assets.py --dest /opt/isaac-assets [--dry-run]

By default: the warehouse scene, the Nova Carter nav sample and robot, and the folder of every asset in site/catalog.json
(textures included). Files already present with the same size are skipped. Mount <dest>/Assets/Isaac/6.1 as /opt/assets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
VERSION = "6.1"
DEFAULT_PREFIXES = [
    "Isaac/Environments/Simple_Warehouse/",  # the base stage the renderer composes under the site layer
    "Isaac/Samples/Replicator/OmniGraph/",  # nova_carter_nav_only.usd, which simulate_lane drives
    "Isaac/Robots/NVIDIA/NovaCarter/",  # the robot that sample references
]
CATALOG = Path(__file__).resolve().parent.parent / "site" / "catalog.json"


def list_objects(client: httpx.Client, prefix: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        url = BUCKET + "?list-type=2&max-keys=1000&prefix=" + urllib.parse.quote(prefix) + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        text = client.get(url, timeout=60).text
        for m in re.finditer(r"<Contents><Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", text, re.S):
            out.append((m.group(1), int(m.group(2))))
        if "<IsTruncated>true</IsTruncated>" not in text:
            return out
        token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text).group(1)


def fetch(client: httpx.Client, key: str, size: int, dest: Path) -> bool:
    local = dest / key
    if local.exists() and local.stat().st_size == size:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    for attempt in range(4):
        try:
            with client.stream("GET", BUCKET + urllib.parse.quote(key), timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            tmp.replace(local)
            return True
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise RuntimeError(f"{key}: {e}") from e
            time.sleep(2 * (attempt + 1))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="local root; files land at <dest>/Assets/Isaac/<version>/...")
    ap.add_argument("--prefix", action="append", default=[], help="extra folder under Isaac/<version>/ to mirror (repeatable)")
    ap.add_argument("--version", default=VERSION)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="list what would be mirrored and its size, download nothing")
    args = ap.parse_args()
    dest = Path(args.dest)
    folders = set(DEFAULT_PREFIXES) | set(args.prefix)
    folders |= {a["usd"].rsplit("/", 1)[0] + "/" for a in json.loads(CATALOG.read_text())["assets"]}
    prefixes = sorted(f"Assets/Isaac/{args.version}/{f}" for f in folders)
    t0 = time.time()
    with httpx.Client(headers={"User-Agent": "durable-simulation-agent-mirror/0.1"}) as client:
        objs = sorted({o for p in prefixes for o in list_objects(client, p)})
        print(f"{len(objs)} objects, {sum(s for _, s in objs) / 1e6:.0f} MB under {len(prefixes)} folders", flush=True)
        if args.dry_run:
            return
        fetched = skipped = 0
        errors: list[str] = []
        with ThreadPoolExecutor(args.workers) as ex:
            for fut in [ex.submit(fetch, client, k, s, dest) for k, s in objs]:
                try:
                    if fut.result():
                        fetched += 1
                    else:
                        skipped += 1
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
                if (fetched + skipped) % 500 == 0:
                    print(f"  {fetched + skipped}/{len(objs)}", flush=True)
    print(json.dumps({"objects": len(objs), "fetched": fetched, "skipped": skipped, "seconds": round(time.time() - t0, 1), "errors": errors[:10],
                      "mount": str(dest / "Assets" / "Isaac" / args.version)}), flush=True)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
