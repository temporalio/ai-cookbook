# ABOUTME: Tests for recipe-lint language detection and check dispatch.

from pathlib import Path

from recipe_lint.dispatch import detect_language, run_checks


def test_detect_language_python_suffix() -> None:
    assert detect_language(Path("foundations/hello_world_openai_responses_python")) == "python"


def test_detect_language_unknown_suffix() -> None:
    assert detect_language(Path("mcp/hello_world_durable_mcp_server")) is None


def test_run_checks_unknown_language_warns_without_crashing(tmp_path: Path) -> None:
    recipe = tmp_path / "some_recipe"
    recipe.mkdir()
    findings = run_checks(recipe)
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].code == "lang-unknown"


def test_run_checks_python_with_no_registered_checks_is_empty(tmp_path: Path) -> None:
    recipe = tmp_path / "thing_python"
    recipe.mkdir()
    assert run_checks(recipe) == []
