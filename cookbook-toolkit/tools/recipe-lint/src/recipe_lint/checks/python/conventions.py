# ABOUTME: AST-based Temporal/Python convention checks for cookbook recipes.
# Scoped strictly to LLM-client and Workflow constructs; never inspects Temporal retry config.

from __future__ import annotations

import ast
from pathlib import Path

from recipe_lint.dispatch import CHECKS
from recipe_lint.findings import Finding

# Rule 1 SCOPE (deliberate): the max_retries check inspects ONLY these LLM/HTTP client
# constructors. `max_retries` is a client-library concern and is a DIFFERENT axis from
# Temporal's Activity retries (RetryPolicy / maximum_attempts on execute_activity). This
# check never reads, matches, or asserts against RetryPolicy, maximum_attempts, or the
# Activity-execution path, and must never flag a workflow's RetryPolicy.
_CLIENT_CONSTRUCTORS = {
    "AsyncOpenAI",
    "OpenAI",
    "AsyncAnthropic",
    "Anthropic",
    "AsyncAzureOpenAI",
    "AzureOpenAI",
}

# Known-retired model names (low false-positive: flag only these, not "unknown" names).
_STALE_MODELS = {
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "text-davinci-003",
    "claude-2",
    "claude-2.1",
    "claude-instant-1",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
}

_TIMEOUT_KWARGS = {"start_to_close_timeout", "schedule_to_close_timeout"}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _kwarg_names(call: ast.Call) -> set[str]:
    return {k.arg for k in call.keywords if k.arg}


def _passes_max_retries_zero(call: ast.Call) -> bool:
    for k in call.keywords:
        if k.arg == "max_retries":
            return isinstance(k.value, ast.Constant) and k.value.value == 0
    return False


def _is_client_connect(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != "connect":
        return False
    receiver = func.value
    if isinstance(receiver, ast.Name):
        return "Client" in receiver.id
    if isinstance(receiver, ast.Attribute):
        return "Client" in receiver.attr
    return False


def findings_for_source(source: str, filename: str) -> list[Finding]:
    """Analyze one Python source string. Exposed for unit tests with code snippets."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding("warning", "parse-error", f"could not parse Python: {exc.msg}", file=filename, line=exc.lineno)]

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _STALE_MODELS:
            findings.append(
                Finding(
                    "warning",
                    "stale-model",
                    f"stale model name '{node.value}' — use a current model",
                    file=filename,
                    line=node.lineno,
                )
            )
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None:
            continue

        # Rule 1: LLM/HTTP client constructors must pass max_retries=0 (clients ONLY).
        if name in _CLIENT_CONSTRUCTORS and not _passes_max_retries_zero(node):
            findings.append(
                Finding(
                    "warning",
                    "client-retries",
                    f"{name}(...) should pass max_retries=0 (Temporal owns retries)",
                    file=filename,
                    line=node.lineno,
                )
            )

        # Rule 2: Pydantic data converter on Client.connect and start_time_skipping.
        if _is_client_connect(node) and "data_converter" not in _kwarg_names(node):
            findings.append(
                Finding(
                    "warning",
                    "data-converter",
                    "Client.connect(...) should pass data_converter=pydantic_data_converter",
                    file=filename,
                    line=node.lineno,
                )
            )
        if name == "start_time_skipping" and "data_converter" not in _kwarg_names(node):
            findings.append(
                Finding(
                    "warning",
                    "data-converter",
                    "start_time_skipping(...) should pass data_converter=pydantic_data_converter",
                    file=filename,
                    line=node.lineno,
                )
            )

        # Rule 3: execute_activity must bound the Activity with a timeout.
        if name in {"execute_activity", "execute_local_activity"} and not (_kwarg_names(node) & _TIMEOUT_KWARGS):
            findings.append(
                Finding(
                    "warning",
                    "no-timeout",
                    f"{name}(...) must set start_to_close_timeout",
                    file=filename,
                    line=node.lineno,
                )
            )

    return findings


def check_code_conventions(recipe_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for py in sorted(recipe_dir.rglob("*.py")):
        if ".venv" in py.parts or "__pycache__" in py.parts:
            continue
        findings.extend(findings_for_source(py.read_text(), str(py.relative_to(recipe_dir))))
    return findings


CHECKS["python"].append(check_code_conventions)
