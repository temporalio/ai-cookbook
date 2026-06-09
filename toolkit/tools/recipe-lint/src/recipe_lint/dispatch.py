# ABOUTME: Language detection and check dispatch for recipe-lint.
# Detects a recipe's language by directory-name suffix and runs the registered checks.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from recipe_lint.findings import Finding

CheckFn = Callable[[Path], list[Finding]]

# Language -> ordered list of checks. Check modules (Steps 8-10) append to this.
CHECKS: dict[str, list[CheckFn]] = {"python": []}

_SUFFIX_LANGUAGE = {
    "_python": "python",
    "_typescript": "typescript",
    "_go": "go",
}


def detect_language(recipe_dir: Path) -> str | None:
    """Resolve a recipe's language from its directory-name suffix, or None."""
    name = recipe_dir.name
    for suffix, language in _SUFFIX_LANGUAGE.items():
        if name.endswith(suffix):
            return language
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
                    "(expected a '_python' suffix)"
                ),
                file=str(recipe_dir),
            )
        ]

    findings: list[Finding] = []
    for check in CHECKS.get(language, []):
        findings.extend(check(recipe_dir))
    return findings
