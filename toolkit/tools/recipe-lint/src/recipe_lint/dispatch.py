# ABOUTME: Language detection and check dispatch for recipe-lint.
# Detects a recipe's language by directory-name suffix and runs the registered checks.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from recipe_lint.findings import Finding

CheckFn = Callable[[Path], list[Finding]]

# Language -> ordered list of checks. Check modules (Steps 8-10) append to this
# on import; _ensure_checks_loaded() imports them lazily to avoid an import cycle.
CHECKS: dict[str, list[CheckFn]] = {"python": []}

_SUFFIX_LANGUAGE = {
    "_python": "python",
    "_typescript": "typescript",
    "_go": "go",
}

_checks_loaded = False


def _ensure_checks_loaded() -> None:
    """Import check modules so they register into CHECKS (idempotent)."""
    global _checks_loaded
    if _checks_loaded:
        return
    # Each check module appends to CHECKS on import. (__init__.py files stay empty
    # per the recipe convention, so import the modules explicitly here.)
    from recipe_lint.checks.python import structure  # noqa: F401

    _checks_loaded = True


def detect_language(recipe_dir: Path) -> str | None:
    """Resolve a recipe's language from its directory-name suffix.

    Falls back to "python" for a recipe directory that has a pyproject.toml but no
    recognized suffix (the MCP recipes), so they are linted as the Python variant.
    """
    name = recipe_dir.name
    for suffix, language in _SUFFIX_LANGUAGE.items():
        if name.endswith(suffix):
            return language
    if (recipe_dir / "pyproject.toml").is_file():
        return "python"
    return None


def run_checks(recipe_dir: Path) -> list[Finding]:
    """Detect the language and run its registered checks. Unknown language warns, never crashes."""
    language = detect_language(recipe_dir)
    if language is None:
        return [
            Finding(
                severity="warning",
                code="lang-unknown",
                message=(
                    f"could not detect language from directory name '{recipe_dir.name}' "
                    "(expected a '_python' suffix or a pyproject.toml)"
                ),
                file=str(recipe_dir),
            )
        ]

    _ensure_checks_loaded()
    findings: list[Finding] = []
    for check in CHECKS.get(language, []):
        findings.extend(check(recipe_dir))
    return findings
