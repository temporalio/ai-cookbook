# ABOUTME: Quality-bar check for recipe-lint: runs ruff against a recipe via the
# toolkit-provided config. Degrades gracefully when ruff is not installed.
#
# mypy is intentionally NOT run across the corpus here: strict mypy needs each recipe's
# installed dependencies, and the LLM SDKs are largely untyped, so it produces noise
# rather than signal. mypy --strict remains the bar for toolkit code (via its own
# pyproject) and for new recipes through their own CI.

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from recipe_lint.dispatch import CHECKS
from recipe_lint.findings import Finding

_RUFF_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "ruff.toml"


def check_ruff(recipe_dir: Path) -> list[Finding]:
    ruff = shutil.which("ruff")
    if ruff is None:
        return [Finding("warning", "ruff-missing", "ruff not installed; skipped the ruff quality check")]

    cmd = [ruff, "check", "--quiet", "--output-format", "json"]
    if _RUFF_CONFIG.is_file():
        cmd += ["--config", str(_RUFF_CONFIG)]
    cmd.append(str(recipe_dir))

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    out = proc.stdout.strip()
    if not out:
        return []
    try:
        diagnostics = json.loads(out)
    except json.JSONDecodeError:
        return []

    findings: list[Finding] = []
    recipe_abs = recipe_dir.resolve()
    for d in diagnostics:
        code = d.get("code") or "ruff"
        message = d.get("message", "")
        filename = d.get("filename", "")
        try:
            rel = str(Path(filename).resolve().relative_to(recipe_abs))
        except ValueError:
            rel = filename
        row = (d.get("location") or {}).get("row")
        findings.append(Finding("warning", f"ruff:{code}", message, file=rel, line=row))
    return findings


CHECKS["python"].append(check_ruff)
