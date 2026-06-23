# ABOUTME: Structural, layout, naming, and link checks for Python cookbook recipes.
# Path- and regex-based (no AST); registers into the recipe-lint python dispatch.

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from recipe_lint.dispatch import CHECKS
from recipe_lint.findings import Finding

# Top-level .py files that are legitimate entrypoints (not "stray").
_ALLOWED_TOPLEVEL = {"worker.py", "start_workflow.py", "send_approval.py"}

# The stray-entry rule targets *duplicate entrypoints* (e.g. hello_world.py,
# claude_test.py) that should be worker.py/start_workflow.py instead. It must NOT
# flag legitimate top-level helper modules (e.g. context_window.py, models.py),
# so a top-level file is only "stray" when its content looks like an entrypoint.
_ENTRYPOINT_MARKERS = (
    '__name__ == "__main__"',
    "__name__ == '__main__'",
    "asyncio.run(",
    "execute_workflow",
    "Worker(",
)


def _is_mcp(recipe_dir: Path) -> bool:
    """MCP recipes are a sanctioned variant: a server entrypoint instead of start_workflow.py."""
    return recipe_dir.parent.name == "mcp" or (recipe_dir / "mcp_servers").is_dir()


def _slug(recipe_dir: Path) -> str:
    name = recipe_dir.name
    base = name[: -len("_python")] if name.endswith("_python") else name
    return base.replace("_", "-")


def check_required_files(recipe_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for fname in ("pyproject.toml", "README.md", "worker.py"):
        if not (recipe_dir / fname).is_file():
            findings.append(Finding("error", "missing-file", f"required file missing: {fname}"))
    if _is_mcp(recipe_dir):
        if not (recipe_dir / "mcp_servers").is_dir():
            findings.append(Finding("warning", "mcp-no-server", "MCP recipe should have an mcp_servers/ directory"))
    elif not (recipe_dir / "start_workflow.py").is_file():
        findings.append(Finding("error", "missing-file", "required file missing: start_workflow.py"))
    return findings


def check_required_dirs(recipe_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for dname in ("activities", "workflows"):
        if not (recipe_dir / dname).is_dir():
            findings.append(Finding("warning", "missing-dir", f"expected directory missing: {dname}/"))
    tests = recipe_dir / "tests"
    if not tests.is_dir() or not any(tests.glob("test_*.py")):
        findings.append(Finding("error", "no-tests", "tests/ is required and must contain test_*.py files"))
    return findings


def check_empty_init(recipe_dir: Path) -> list[Finding]:
    """If an __init__.py exists, it must be empty (no code)."""
    findings: list[Finding] = []
    for init in recipe_dir.rglob("__init__.py"):
        if ".venv" in init.parts:
            continue
        if init.read_text().strip():
            rel = init.relative_to(recipe_dir)
            findings.append(
                Finding(
                    "warning",
                    "init-not-empty",
                    "__init__.py must be empty (move code to a named module)",
                    file=str(rel),
                )
            )
    return findings


def check_package_name(recipe_dir: Path) -> list[Finding]:
    pyproject = recipe_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        data = tomllib.loads(pyproject.read_text())
    except (tomllib.TOMLDecodeError, OSError) as exc:
        return [Finding("warning", "pyproject-parse", f"could not parse pyproject.toml: {exc}", file="pyproject.toml")]
    name = data.get("project", {}).get("name")
    expected = f"cookbook-{_slug(recipe_dir)}-python"
    if name != expected:
        return [
            Finding("warning", "package-name", f"package name '{name}' should be '{expected}'", file="pyproject.toml")
        ]
    return []


def check_python_floor(recipe_dir: Path) -> list[Finding]:
    """requires-python must declare a `>=3.10` floor; the upper bound is not constrained."""
    pyproject = recipe_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        data = tomllib.loads(pyproject.read_text())
    except (tomllib.TOMLDecodeError, OSError):
        return []  # check_package_name reports the parse error
    req = data.get("project", {}).get("requires-python")
    if not req:
        return [
            Finding(
                "warning",
                "python-floor",
                "pyproject.toml is missing requires-python; declare a `>=3.10` floor",
                file="pyproject.toml",
            )
        ]
    lower = next((c.strip() for c in str(req).split(",") if c.strip().startswith(">=")), None)
    if lower is None or lower.replace(" ", "") != ">=3.10":
        return [
            Finding(
                "warning",
                "python-floor",
                f"requires-python floor should be `>=3.10` (found '{req}')",
                file="pyproject.toml",
            )
        ]
    return []


def check_task_queue(recipe_dir: Path) -> list[Finding]:
    expected = f"{_slug(recipe_dir)}-task-queue"
    findings: list[Finding] = []
    for fname in ("worker.py", "start_workflow.py"):
        f = recipe_dir / fname
        if f.is_file() and expected not in f.read_text():
            findings.append(
                Finding("warning", "task-queue", f"expected task queue '{expected}' not found in {fname}", file=fname)
            )
    return findings


def check_stray_entry(recipe_dir: Path) -> list[Finding]:
    """Flag a *duplicate entrypoint* at the top level, not a helper module.

    Top-level helper modules (e.g. context_window.py) are allowed; only a
    top-level file whose content looks like a worker/starter entrypoint is
    flagged, since worker.py and start_workflow.py are the only entrypoints.
    """
    findings: list[Finding] = []
    for f in recipe_dir.glob("*.py"):
        if f.name in _ALLOWED_TOPLEVEL:
            continue
        if any(marker in f.read_text() for marker in _ENTRYPOINT_MARKERS):
            findings.append(
                Finding(
                    "warning",
                    "stray-entry",
                    f"top-level '{f.name}' looks like an extra entrypoint; "
                    "use worker.py and start_workflow.py as the only entrypoints",
                    file=f.name,
                )
            )
    return findings


def check_readme_frontmatter(recipe_dir: Path) -> list[Finding]:
    readme = recipe_dir / "README.md"
    if not readme.is_file():
        return []
    text = readme.read_text()
    findings: list[Finding] = []
    if not re.match(r"^\s*<!--", text):
        findings.append(
            Finding("warning", "frontmatter-missing", "README has no front-matter comment block", file="README.md")
        )
    body = re.sub(r"^\s*<!--.*?-->", "", text, count=1, flags=re.S).lstrip()
    if not re.match(r"^#\s+\S", body):
        findings.append(
            Finding("warning", "h1-missing", "README has no H1 title after the front matter", file="README.md")
        )
    return findings


_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"')


def check_links(recipe_dir: Path) -> list[Finding]:
    readme = recipe_dir / "README.md"
    if not readme.is_file():
        return []
    text = readme.read_text()
    findings: list[Finding] = []
    for m in _LINK_RE.finditer(text):
        target = m.group(1).split()[0].split("#")[0]
        if not target or target.startswith(("http://", "https://", "mailto:", "/")):
            continue
        if not (recipe_dir / target).exists():
            findings.append(
                Finding("warning", "broken-link", f"README link target not found: {target}", file="README.md")
            )
    for m in _IMG_RE.finditer(text):
        target = m.group(1)
        if target.startswith(("http://", "https://", "/")):
            continue
        if not (recipe_dir / target).exists():
            findings.append(Finding("warning", "broken-asset", f"README image not found: {target}", file="README.md"))
    return findings


CHECKS["python"].extend(
    [
        check_required_files,
        check_required_dirs,
        check_empty_init,
        check_package_name,
        check_python_floor,
        check_task_queue,
        check_stray_entry,
        check_readme_frontmatter,
        check_links,
    ]
)
