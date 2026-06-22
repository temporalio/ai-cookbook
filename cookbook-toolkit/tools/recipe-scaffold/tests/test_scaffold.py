# ABOUTME: Tests that recipe-scaffold renders the expected files with substituted values.

import ast

import pytest

from recipe_scaffold.card import context_from_fields
from recipe_scaffold.scaffold import render


def _ctx():  # type: ignore[no-untyped-def]
    return context_from_fields(
        name="example-thing",
        category="foundations",
        title="Example Thing",
        description="An example recipe.",
        priority=500,
        providers=["openai"],
    )


def test_render_writes_expected_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dest = tmp_path / "foundations" / "example_thing_python"
    written = render(_ctx(), dest)
    names = {p.relative_to(dest).as_posix() for p in written}
    assert names == {
        "pyproject.toml",
        "README.md",
        "worker.py",
        "start_workflow.py",
        "activities/llm_call.py",
        "workflows/recipe_workflow.py",
        "tests/test_workflow.py",
    }


def test_rendered_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dest = tmp_path / "foundations" / "example_thing_python"
    render(_ctx(), dest)
    pyproject = (dest / "pyproject.toml").read_text()
    assert 'name = "cookbook-example-thing-python"' in pyproject
    assert "openai>=1.40.0" in pyproject
    assert "example-thing-task-queue" in (dest / "worker.py").read_text()
    readme = (dest / "README.md").read_text()
    assert "# Example Thing" in readme
    assert "tags: [foundations, python, openai]" in readme


def test_rendered_python_parses(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dest = tmp_path / "f" / "e_python"
    render(
        context_from_fields(name="e", category="foundations", title="E", description="d.", priority=1),
        dest,
    )
    for py in dest.rglob("*.py"):
        ast.parse(py.read_text())


def test_force_required_for_nonempty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dest = tmp_path / "foundations" / "example_thing_python"
    render(_ctx(), dest)
    with pytest.raises(FileExistsError):
        render(_ctx(), dest)
    render(_ctx(), dest, force=True)
