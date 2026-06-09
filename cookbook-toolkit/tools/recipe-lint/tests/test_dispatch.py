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


def test_run_checks_python_runs_registered_checks(tmp_path: Path) -> None:
    # An empty python recipe dir triggers structural findings (missing files, no tests).
    recipe = tmp_path / "thing_python"
    recipe.mkdir()
    findings = run_checks(recipe)
    assert any(f.code == "no-tests" for f in findings)


def test_detect_language_pyproject_fallback(tmp_path: Path) -> None:
    # A dir without a recognized suffix but with a pyproject.toml is the python (MCP) variant.
    recipe = tmp_path / "weather_server"
    recipe.mkdir()
    (recipe / "pyproject.toml").write_text("[project]\nname='x'\n")
    from recipe_lint.dispatch import detect_language

    assert detect_language(recipe) == "python"
