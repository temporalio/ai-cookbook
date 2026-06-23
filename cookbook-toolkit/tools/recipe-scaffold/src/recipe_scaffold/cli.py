# ABOUTME: CLI for recipe-scaffold. Scaffolds a recipe from a proposal card or explicit fields.
# Writes {category}/{name}_python under the target root and prints the files created.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recipe_scaffold.card import (
    CardError,
    ScaffoldContext,
    context_from_card,
    context_from_fields,
    load_card,
)
from recipe_scaffold.scaffold import render


def _resolve_context(args: argparse.Namespace) -> ScaffoldContext:
    if args.card:
        return context_from_card(load_card(Path(args.card)))
    missing = [f for f in ("name", "category", "title", "description", "priority") if getattr(args, f) is None]
    if missing:
        raise SystemExit(f"error: without --card these flags are required: {', '.join('--' + m for m in missing)}")
    return context_from_fields(
        name=args.name,
        category=args.category,
        title=args.title,
        description=args.description,
        priority=args.priority,
        providers=args.provider or None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recipe-scaffold",
        description="Deterministically scaffold a cookbook recipe from a proposal card or explicit fields.",
    )
    parser.add_argument("--card", help="Path to a proposal card (YAML). Overrides the field flags below.")
    parser.add_argument("--name", help="kebab-case recipe name.")
    parser.add_argument("--category", choices=["agents", "foundations", "deep_research", "mcp"])
    parser.add_argument("--provider", action="append", choices=["openai", "anthropic", "litellm"])
    parser.add_argument("--title")
    parser.add_argument("--description")
    parser.add_argument("--priority", type=int)
    parser.add_argument("--into", default=".", help="Root to scaffold under (default: cwd / repo root).")
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty target directory.")
    args = parser.parse_args(argv)

    try:
        ctx = _resolve_context(args)
    except CardError as exc:
        raise SystemExit(f"error: {exc}") from exc
    dest = Path(args.into) / ctx.category / ctx.dir_name
    written = render(ctx, dest, force=args.force)

    print(f"Scaffolded {ctx.package} at {dest}")
    for path in written:
        print(f"  {path}")
    print("\nNext: fill the stubbed Activity logic + README prose, then run recipe-lint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
