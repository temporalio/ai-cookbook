# ABOUTME: Finding data model, report formatter, and exit-code rule for recipe-lint.
# A Finding is one lint result; warnings never fail the run, errors do (lint-not-block).

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Finding:
    """One lint result for a recipe."""

    severity: Severity
    code: str
    message: str
    file: str | None = None
    line: int | None = None


def exit_code(findings: list[Finding]) -> int:
    """Lint-not-block: any error → nonzero; warnings-only or empty → zero."""
    return 1 if any(f.severity == "error" for f in findings) else 0


def format_report(findings: list[Finding], recipe_dir: str) -> str:
    """Render findings as a human-readable report, errors before warnings."""
    if not findings:
        return f"recipe-lint: {recipe_dir}\n  no findings"

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]

    lines = [f"recipe-lint: {recipe_dir}"]
    for label, group, mark in (("error", errors, "x"), ("warning", warnings, "!")):
        for f in group:
            location = ""
            if f.file:
                location = f" ({f.file}:{f.line})" if f.line else f" ({f.file})"
            lines.append(f"  {mark} {label} [{f.code}] {f.message}{location}")
    lines.append(f"  -- {len(errors)} error(s), {len(warnings)} warning(s)")
    return "\n".join(lines)
