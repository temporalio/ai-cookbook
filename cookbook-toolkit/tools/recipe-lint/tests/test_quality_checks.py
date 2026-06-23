# ABOUTME: Tests for recipe-lint's ruff quality check (violation, clean, missing-tool).

from pathlib import Path

import recipe_lint.checks.python.quality as quality
from recipe_lint.checks.python.quality import check_ruff


def test_ruff_flags_a_violation(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("import os\n")  # F401 unused import
    findings = check_ruff(tmp_path)
    assert any(f.code.startswith("ruff:") for f in findings), findings


def test_ruff_clean_file_has_no_findings(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text('"""Doc."""\n\nx = 1\nprint(x)\n')
    assert check_ruff(tmp_path) == []


def test_ruff_honors_discovered_config(tmp_path: Path) -> None:
    # recipe-lint relies on ruff discovering the repo-root ruff.toml by walking
    # up from the recipe, rather than passing --config. `N` (pep8-naming) is not
    # in ruff's default select, so an N-code finding proves the discovered
    # config's `select` was applied instead of ruff's built-in defaults.
    (tmp_path / "ruff.toml").write_text('[lint]\nselect = ["N"]\n')
    (tmp_path / "mod.py").write_text("def BadName():\n    return 1\n")
    findings = check_ruff(tmp_path)
    assert any(f.code.startswith("ruff:N") for f in findings), findings


def test_ruff_missing_degrades_gracefully(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(quality.shutil, "which", lambda _name: None)
    findings = check_ruff(tmp_path)
    assert len(findings) == 1
    assert findings[0].code == "ruff-missing"
