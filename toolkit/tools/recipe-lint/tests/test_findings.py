# ABOUTME: Tests for recipe-lint's Finding model, exit-code rule, and report formatter.

from recipe_lint.findings import Finding, exit_code, format_report


def test_exit_code_error_is_nonzero() -> None:
    assert exit_code([Finding("error", "x", "boom")]) == 1


def test_exit_code_warnings_only_is_zero() -> None:
    assert exit_code([Finding("warning", "x", "meh")]) == 0


def test_exit_code_empty_is_zero() -> None:
    assert exit_code([]) == 0


def test_format_report_groups_errors_before_warnings_with_location() -> None:
    findings = [
        Finding("warning", "w1", "a warning", file="foo.py", line=3),
        Finding("error", "e1", "an error", file="bar.py"),
    ]
    report = format_report(findings, "some/recipe")
    assert "some/recipe" in report
    assert report.index("an error") < report.index("a warning")
    assert "foo.py:3" in report
    assert "bar.py" in report


def test_format_report_empty_says_no_findings() -> None:
    assert "no findings" in format_report([], "some/recipe")
