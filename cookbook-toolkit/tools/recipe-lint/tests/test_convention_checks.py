# ABOUTME: Tests for recipe-lint's AST code-convention checks (max_retries, data converter,
# timeouts, stale models) and the critical max_retries/RetryPolicy separation.

from recipe_lint.checks.python.conventions import findings_for_source


def _codes(source: str) -> set[str]:
    return {f.code for f in findings_for_source(source, "x.py")}


def test_client_without_max_retries_is_flagged() -> None:
    assert "client-retries" in _codes("client = AsyncAnthropic()")


def test_client_with_max_retries_zero_is_clean() -> None:
    assert "client-retries" not in _codes("client = AsyncAnthropic(max_retries=0)")


def test_client_with_nonzero_max_retries_is_flagged() -> None:
    assert "client-retries" in _codes("client = AsyncOpenAI(max_retries=3)")


def test_client_connect_without_data_converter_is_flagged() -> None:
    assert "data-converter" in _codes('c = await Client.connect("localhost:7233")')


def test_client_connect_with_data_converter_is_clean() -> None:
    src = 'c = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)'
    assert "data-converter" not in _codes(src)


def test_execute_activity_without_timeout_is_flagged() -> None:
    assert "no-timeout" in _codes("await workflow.execute_activity(classify, req)")


def test_execute_activity_with_timeout_is_clean() -> None:
    src = "await workflow.execute_activity(classify, req, start_to_close_timeout=timedelta(seconds=30))"
    assert "no-timeout" not in _codes(src)


def test_stale_model_literal_is_flagged() -> None:
    assert "stale-model" in _codes('model = "claude-3-opus-20240229"')


def test_current_model_literal_is_clean() -> None:
    assert "stale-model" not in _codes('model = "claude-sonnet-4-6"')


def test_retry_policy_is_never_flagged_as_client_retries() -> None:
    # CRITICAL: max_retries (client) and RetryPolicy/maximum_attempts (Temporal Activity
    # retries) are different axes. A workflow's RetryPolicy must NEVER be flagged, and the
    # max_retries check must not touch the Activity-execution path.
    source = (
        "await workflow.execute_activity(\n"
        "    classify, req,\n"
        "    start_to_close_timeout=timedelta(seconds=30),\n"
        "    retry_policy=RetryPolicy(maximum_attempts=3),\n"
        ")\n"
        "client = AsyncAnthropic(max_retries=0)\n"
    )
    codes = _codes(source)
    assert "client-retries" not in codes
    assert "no-timeout" not in codes
    # No finding should reference maximum_attempts / RetryPolicy at all.
    msgs = " ".join(f.message for f in findings_for_source(source, "x.py"))
    assert "maximum_attempts" not in msgs
    assert "RetryPolicy" not in msgs
