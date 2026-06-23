# ABOUTME: Command-line entry point for recipe-lint.
# Runs the registered checks for a recipe directory and prints a text or JSON report.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recipe_lint.checks.python.quality import fix_style
from recipe_lint.dispatch import run_checks
from recipe_lint.findings import Finding, exit_code, format_report


def _as_dict(finding: Finding) -> dict[str, object]:
    return {
        "severity": finding.severity,
        "code": finding.code,
        "message": finding.message,
        "file": finding.file,
        "line": finding.line,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recipe-lint",
        description="Lint a Temporal AI Cookbook recipe against the cookbook conventions.",
    )
    parser.add_argument("recipe_dir", type=Path, help="Path to the recipe directory.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply ruff autofixes and formatting (using the toolkit config) before reporting.",
    )
    args = parser.parse_args(argv)

    recipe_dir: Path = args.recipe_dir
    if not recipe_dir.is_dir():
        parser.error(f"not a directory: {recipe_dir}")

    if args.fix:
        for line in fix_style(recipe_dir):
            print(f"fix: {line}")

    findings = run_checks(recipe_dir)

    if args.format == "json":
        print(json.dumps([_as_dict(f) for f in findings], indent=2))
    else:
        print(format_report(findings, str(recipe_dir)))

    return exit_code(findings)


if __name__ == "__main__":
    sys.exit(main())
