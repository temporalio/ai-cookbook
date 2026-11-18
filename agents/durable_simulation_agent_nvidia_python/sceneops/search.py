"""sceneops.search: the TOOLS-sandbox entrypoint. One HTTPS call to USD Search; a denied egress is a result, not an error.

    python3 -I -m sceneops.search --host search.simready.omniverse.nvidia.com --query "packing table" [--k 8]

Prints one JSON line: {"denied": false, "hits": [{"url", "score"}, ...]} or {"denied": true, "error": "..."} (exit 3).
OpenShell denies an egress at the socket (permission denied, connection error) or at the HTTP layer (403 policy_denied);
both are reported as denied. The catalog fallback and the mapping of hits to catalog ids happen outside the sandbox.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sceneops.search")
    ap.add_argument("--host", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=20.0)
    a = ap.parse_args(argv)
    req = urllib.request.Request(f"https://{a.host}/search", data=json.dumps({"description": a.query, "limit": a.k}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            hits = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode(errors="replace")
        denied = e.code == 403 or "policy_denied" in body
        print(json.dumps({"denied": denied, "error": f"HTTP {e.code}: {body}"}))
        sys.exit(3 if denied else 0)
    except OSError as e:  # [Errno 13] Permission denied, connection refused, name resolution: no policy rule for this host
        print(json.dumps({"denied": True, "error": str(e)[:200]}))
        sys.exit(3)
    out = [{"url": h.get("url", ""), "score": h.get("score")} for h in hits if isinstance(h, dict)] if isinstance(hits, list) else []
    print(json.dumps({"denied": False, "hits": out}))


if __name__ == "__main__":
    main()
