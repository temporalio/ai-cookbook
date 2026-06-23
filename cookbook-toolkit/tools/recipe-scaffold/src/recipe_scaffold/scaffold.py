# ABOUTME: Render the recipe-skeleton Jinja templates into a target recipe directory.
# Deterministic: same context always yields the same lint-clean recipe scaffold.

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from recipe_scaffold.card import ScaffoldContext

# cookbook-toolkit/templates/recipe-skeleton, from this module's location:
# .../tools/recipe-scaffold/src/recipe_scaffold/scaffold.py
_TEMPLATES_DIR = Path(__file__).resolve().parents[4] / "templates" / "recipe-skeleton"


def render(ctx: ScaffoldContext, dest: Path, *, force: bool = False) -> list[Path]:
    """Render every *.j2 template under the skeleton into `dest`. Returns the files written."""
    if dest.exists() and any(dest.iterdir()) and not force:
        raise FileExistsError(f"{dest} already exists and is not empty (pass force=True to overwrite)")

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    values = ctx.as_template_values()

    written: list[Path] = []
    for template_path in sorted(_TEMPLATES_DIR.rglob("*.j2")):
        rel = template_path.relative_to(_TEMPLATES_DIR)
        out_path = dest / rel.with_suffix("")  # strip the .j2 suffix
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = env.get_template(rel.as_posix()).render(**values)
        out_path.write_text(rendered)
        written.append(out_path)
    return written
