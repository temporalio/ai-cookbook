# ABOUTME: Quality-bar check for recipe-lint: runs ruff against a recipe using the
# repo-root ruff.toml that ruff discovers by walking up. Degrades gracefully when
# ruff is not installed.
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


def _find_ruff() -> str | None:
    """Resolve the ruff executable. Pinned as a runtime dep, so it is on PATH
    inside the tool's uv environment; falls back to a system ruff."""
    return shutil.which("ruff")


def fix_style(recipe_dir: Path) -> list[str]:
    """Apply ruff's autofixes and formatter to a recipe.

    No --config: ruff discovers the repo-root ruff.toml by walking up from the
    recipe, the same config a contributor's editor or local ruff resolves.

    Returns human-readable summary lines. Run by `recipe-lint --fix` before the
    report, so the same ruff and config that flag style issues also fix them.
    """
    ruff = _find_ruff()
    if ruff is None:
        return ["ruff not installed; skipped --fix"]

    summary: list[str] = []
    check = subprocess.run(  # noqa: S603
        [ruff, "check", "--fix", "--quiet", str(recipe_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    outcome = "applied fixes" if check.returncode == 0 else "fixed what it could; some findings remain"
    summary.append(f"ruff check --fix: {outcome}")
    fmt = subprocess.run(  # noqa: S603
        [ruff, "format", "--quiet", str(recipe_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    summary.append("ruff format: " + ("formatted" if fmt.returncode == 0 else "format reported an error"))
    return summary


def check_ruff(recipe_dir: Path) -> list[Finding]:
    ruff = _find_ruff()
    if ruff is None:
        return [Finding("warning", "ruff-missing", "ruff not installed; skipped the ruff quality check")]

    cmd = [ruff, "check", "--quiet", "--output-format", "json", str(recipe_dir)]

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


def check_format(recipe_dir: Path) -> list[Finding]:
    """Report files that are not ruff-formatted. `recipe-lint --fix` resolves these."""
    ruff = _find_ruff()
    if ruff is None:
        return []

    proc = subprocess.run(  # noqa: S603
        [ruff, "format", "--check", str(recipe_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return []

    recipe_abs = recipe_dir.resolve()
    findings: list[Finding] = []
    for line in proc.stdout.splitlines():
        if not line.startswith("Would reformat:"):
            continue
        filename = line.split(":", 1)[1].strip()
        try:
            rel = str(Path(filename).resolve().relative_to(recipe_abs))
        except ValueError:
            rel = filename
        findings.append(
            Finding("warning", "ruff-format", "file is not formatted; run `recipe-lint --fix`", file=rel)
        )
    if not findings:
        findings.append(Finding("warning", "ruff-format", "some files are not formatted; run `recipe-lint --fix`"))
    return findings


CHECKS["python"].append(check_ruff)
CHECKS["python"].append(check_format)
