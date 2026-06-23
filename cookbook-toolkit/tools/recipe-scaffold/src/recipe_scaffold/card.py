# ABOUTME: Load + validate a recipe proposal card and derive the deterministic scaffold context.
# The card's `recipe:` block (validated against card-schema.json) maps to a ScaffoldContext.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

# card-schema.json lives in the skill references; this tool is at
# cookbook-toolkit/tools/recipe-scaffold/src/recipe_scaffold/card.py
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[4] / "skills" / "recipe-writing" / "references" / "card-schema.json"
)

_PROVIDER_DEPS = {
    "openai": "openai>=1.40.0",
    "anthropic": "anthropic>=0.40.0",
    "litellm": "litellm>=1.40.0",
}

_PROVIDER_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
    "litellm": "gpt-4o-mini",
}


def _pascal(snake: str) -> str:
    return "".join(part.title() for part in snake.replace("-", "_").split("_") if part)


@dataclass(frozen=True)
class ScaffoldContext:
    """Everything the Jinja templates need, derived from a card's `recipe:` block."""

    slug: str
    category: str
    language: str
    title: str
    description: str
    priority: int
    providers: list[str]
    workflow_class: str
    activity_func: str

    @property
    def package(self) -> str:
        return f"cookbook-{self.slug}-python"

    @property
    def task_queue(self) -> str:
        return f"{self.slug}-task-queue"

    @property
    def dir_name(self) -> str:
        return f"{self.slug.replace('-', '_')}_python"

    @property
    def activity_module(self) -> str:
        # Fixed module name so worker/test imports are stable regardless of the function name.
        return "llm_call"

    @property
    def request_class(self) -> str:
        return f"{_pascal(self.activity_func)}Request"

    @property
    def default_model(self) -> str:
        return _PROVIDER_MODEL.get(self.primary_provider or "", "gpt-4o-mini")

    @property
    def provider_deps(self) -> list[str]:
        return [_PROVIDER_DEPS[p] for p in self.providers]

    @property
    def primary_provider(self) -> str | None:
        return self.providers[0] if self.providers else None

    @property
    def tags(self) -> list[str]:
        return [self.category, self.language, *self.providers]

    def as_template_values(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "category": self.category,
            "language": self.language,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "providers": self.providers,
            "workflow_class": self.workflow_class,
            "activity_func": self.activity_func,
            "activity_module": self.activity_module,
            "request_class": self.request_class,
            "package": self.package,
            "task_queue": self.task_queue,
            "dir_name": self.dir_name,
            "default_model": self.default_model,
            "provider_deps": self.provider_deps,
            "primary_provider": self.primary_provider,
            "tags": self.tags,
        }


def _schema() -> dict[str, Any]:
    schema: dict[str, Any] = json.loads(_SCHEMA_PATH.read_text())
    return schema


class CardError(Exception):
    """A proposal card failed to load or validate, with a human-facing message."""


def _format_path(parts: Any) -> str:
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out or "(root)"


def _explain_validation_error(path: Path, exc: jsonschema.ValidationError) -> str:
    loc = _format_path(exc.absolute_path)
    # The most common authoring mistake: a prose value with an unquoted colon
    # (e.g. a notes item "Test strategy: ...") parses as a YAML mapping, so the
    # schema sees a dict where it wants a string.
    if exc.validator == "type" and exc.validator_value == "string" and isinstance(exc.instance, dict):
        keys = ", ".join(repr(k) for k in exc.instance)
        return (
            f"{path}: '{loc}' parsed as a YAML mapping (keys: {keys}), but it must be a string. "
            "This usually means an unquoted colon in the value. Wrap it in double quotes "
            'or use a block scalar, e.g.  - "Test strategy: mock the LLM"  or  - >-'
        )
    return f"{path}: invalid card at '{loc}': {exc.message}"


def load_card(path: Path) -> dict[str, Any]:
    """Parse a YAML card and validate it against card-schema.json.

    Raises CardError with a human-facing message on malformed YAML or schema
    violations, so the CLI can print guidance instead of a stack trace.
    """
    try:
        data: dict[str, Any] = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise CardError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise CardError(f"{path}: expected a YAML mapping with a 'recipe:' block.")
    try:
        jsonschema.validate(data, _schema())
    except jsonschema.ValidationError as exc:
        raise CardError(_explain_validation_error(path, exc)) from exc
    return data


def context_from_card(card: dict[str, Any]) -> ScaffoldContext:
    return _context(card["recipe"])


def context_from_fields(
    *,
    name: str,
    category: str,
    title: str,
    description: str,
    priority: int,
    language: str = "python",
    providers: list[str] | None = None,
) -> ScaffoldContext:
    """Build a context directly (used by new-recipe's interactive flow), validating via the schema."""
    recipe: dict[str, Any] = {
        "name": name,
        "category": category,
        "language": language,
        "title": title,
        "description": description,
        "priority": priority,
    }
    if providers:
        recipe["provider"] = providers
    jsonschema.validate({"recipe": recipe}, _schema())
    return _context(recipe)


def _context(recipe: dict[str, Any]) -> ScaffoldContext:
    components = recipe.get("components", {})
    activities = components.get("activities") or ["call_llm"]
    return ScaffoldContext(
        slug=recipe["name"],
        category=recipe["category"],
        language=recipe["language"],
        title=recipe["title"],
        description=recipe["description"],
        priority=recipe["priority"],
        providers=list(recipe.get("provider", [])),
        workflow_class=components.get("workflow_class", "RecipeWorkflow"),
        activity_func=activities[0],
    )
