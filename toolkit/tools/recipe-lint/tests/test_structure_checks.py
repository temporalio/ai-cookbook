# ABOUTME: Tests for recipe-lint's Python structural/layout/naming/link checks.
# Builds tiny fixture recipe dirs (good + bad variants) under tmp_path.

from pathlib import Path

from recipe_lint.checks.python import structure


def _good_recipe(root: Path, name: str = "thing_python") -> Path:
    d = root / name
    (d / "activities").mkdir(parents=True)
    (d / "workflows").mkdir()
    (d / "tests").mkdir()
    (d / "tests" / "test_it.py").write_text("def test_x() -> None:\n    assert True\n")
    (d / "pyproject.toml").write_text('[project]\nname = "cookbook-thing-python"\n')
    (d / "README.md").write_text("<!--\ndescription: x\ntags: [foundations, python]\npriority: 500\n-->\n\n# Thing\n")
    (d / "worker.py").write_text('task_queue = "thing-task-queue"\n')
    (d / "start_workflow.py").write_text('task_queue = "thing-task-queue"\n')
    return d


def test_clean_recipe_has_no_findings(tmp_path: Path) -> None:
    d = _good_recipe(tmp_path)
    findings = (
        structure.check_required_files(d)
        + structure.check_required_dirs(d)
        + structure.check_package_name(d)
        + structure.check_task_queue(d)
        + structure.check_stray_entry(d)
        + structure.check_readme_frontmatter(d)
        + structure.check_links(d)
    )
    assert findings == [], findings


def test_missing_tests_is_error(tmp_path: Path) -> None:
    d = _good_recipe(tmp_path)
    for f in (d / "tests").glob("*"):
        f.unlink()
    (d / "tests").rmdir()
    findings = structure.check_required_dirs(d)
    assert any(f.severity == "error" and f.code == "no-tests" for f in findings)


def test_wrong_package_name_is_warning(tmp_path: Path) -> None:
    d = _good_recipe(tmp_path)
    (d / "pyproject.toml").write_text('[project]\nname = "cookbook-basic-python"\n')
    findings = structure.check_package_name(d)
    assert any(f.code == "package-name" for f in findings)


def test_broken_readme_link_is_warning(tmp_path: Path) -> None:
    d = _good_recipe(tmp_path)
    (d / "README.md").write_text("<!--\ndescription: x\n-->\n\n# Thing\n\nSee [missing](does/not/exist.py).\n")
    findings = structure.check_links(d)
    assert any(f.code == "broken-link" for f in findings)


def test_non_empty_init_is_warning(tmp_path: Path) -> None:
    d = _good_recipe(tmp_path)
    (d / "activities" / "__init__.py").write_text("from .x import y\n")
    findings = structure.check_empty_init(d)
    assert any(f.code == "init-not-empty" for f in findings)


def test_empty_init_is_clean(tmp_path: Path) -> None:
    d = _good_recipe(tmp_path)
    (d / "activities" / "__init__.py").write_text("")
    assert structure.check_empty_init(d) == []


def test_stray_entry_file_is_warning(tmp_path: Path) -> None:
    d = _good_recipe(tmp_path)
    (d / "hello_world.py").write_text("print('hi')\n")
    findings = structure.check_stray_entry(d)
    assert any(f.code == "stray-entry" for f in findings)


def test_mcp_recipe_does_not_require_start_workflow(tmp_path: Path) -> None:
    mcp = tmp_path / "mcp"
    mcp.mkdir()
    d = _good_recipe(mcp, name="weather_server")
    (d / "start_workflow.py").unlink()
    (d / "mcp_servers").mkdir()
    findings = structure.check_required_files(d)
    assert not any(f.code == "missing-file" for f in findings)
