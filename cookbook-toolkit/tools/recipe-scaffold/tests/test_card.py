# ABOUTME: Tests for proposal-card loading, schema validation, and context derivation.

import pytest
from jsonschema import ValidationError

from recipe_scaffold.card import context_from_card, context_from_fields, load_card


def test_context_from_fields_derives_names() -> None:
    ctx = context_from_fields(
        name="guardrails-hard-rules",
        category="agents",
        title="Guardrails",
        description="A guardrail layer.",
        priority=500,
        providers=["anthropic"],
    )
    assert ctx.package == "cookbook-guardrails-hard-rules-python"
    assert ctx.task_queue == "guardrails-hard-rules-task-queue"
    assert ctx.dir_name == "guardrails_hard_rules_python"
    assert ctx.tags == ["agents", "python", "anthropic"]
    assert ctx.default_model == "claude-sonnet-4-6"
    assert ctx.provider_deps == ["anthropic>=0.40.0"]


def test_context_no_provider() -> None:
    ctx = context_from_fields(name="x-y", category="mcp", title="T", description="D.", priority=1)
    assert ctx.tags == ["mcp", "python"]
    assert ctx.primary_provider is None
    assert ctx.provider_deps == []
    assert ctx.default_model == "gpt-4o-mini"


def test_context_from_card() -> None:
    card = {
        "recipe": {
            "name": "rag-openai",
            "category": "foundations",
            "language": "python",
            "title": "RAG",
            "description": "x.",
            "priority": 700,
            "provider": ["openai"],
        }
    }
    ctx = context_from_card(card)
    assert ctx.slug == "rag-openai"
    assert ctx.provider_deps == ["openai>=1.40.0"]


def test_invalid_name_rejected() -> None:
    with pytest.raises(ValidationError):
        context_from_fields(name="Bad Name", category="agents", title="t", description="d", priority=1)


def test_invalid_category_rejected() -> None:
    with pytest.raises(ValidationError):
        context_from_fields(name="ok", category="nope", title="t", description="d", priority=1)


def test_load_card_validates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    good = tmp_path / "card.yaml"
    good.write_text(
        "recipe:\n  name: x-y\n  category: mcp\n  language: python\n"
        "  title: T\n  description: D.\n  priority: 1\n"
    )
    assert load_card(good)["recipe"]["name"] == "x-y"

    bad = tmp_path / "bad.yaml"
    bad.write_text("recipe:\n  name: x-y\n  category: mcp\n")  # missing required fields
    with pytest.raises(ValidationError):
        load_card(bad)
