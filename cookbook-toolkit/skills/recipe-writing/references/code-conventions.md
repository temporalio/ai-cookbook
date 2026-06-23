# Recipe code conventions

The Python/Temporal rules every recipe follows. Each rule notes its **check type**:

- **mechanical** → enforced by `recipe-lint` (Steps 8-10). Deterministic, AST/parse-based.
- **judgment** → checked by the `recipe-reviewer` agent. Needs reading comprehension.

These are conventions the existing recipes already embody (verified against
`agents/tool_call_openai_python` and `foundations/hello_world_openai_responses_python`).

## Canonical Temporal sources

The rules in this file are the recurring, mostly lint-checkable ones. They are not the whole
of correct Temporal usage. For anything beyond them (determinism, signals, queries, updates,
child workflows, continue-as-new, cancellation, heartbeats, activity timeout and retry
semantics, replay), the **Temporal Developer skill** (`temporal:temporal-developer`) and the
**Temporal Docs MCP** (`temporal-docs`, `https://temporal.mcp.kapa.ai`) are the canonical
source of truth. Consult them instead of relying on memory. Both ship with this toolkit (the
MCP is declared in the plugin manifest; the skill is a required companion, see the toolkit
README). Scout, generate, and review all defer to these sources for Temporal-correctness
judgments: a pattern you cannot confirm against them is a finding, not a pass.

## Durability rules

### 1. LLM/API clients use `max_retries=0`  ·  *mechanical*

Temporal owns retries via the Activity retry policy. Client-side retries interfere with
durable error handling and double-retry.

```python
client = AsyncAnthropic(max_retries=0)   # correct
client = AsyncAnthropic()                 # incorrect, client retries on
```

**Scope.** `max_retries` is *exclusively* an
LLM/HTTP client-library concern (the OpenAI/Anthropic/httpx SDK constructor). It is a
**different axis** from Temporal's Activity retries, which are configured separately via a
`RetryPolicy` / `maximum_attempts` on `execute_activity` and are *expected* to be present.

The `recipe-lint` check for this rule MUST:

- inspect **only** known LLM/HTTP client constructors (e.g. `AsyncOpenAI`, `OpenAI`,
  `AsyncAnthropic`, `Anthropic`, `httpx.Client`/`AsyncClient`) for the `max_retries=0`
  keyword;
- **never** read, match, assert against, or "test" `max_retries` in relation to Temporal's
  Activity retry configuration, `RetryPolicy`, `maximum_attempts`, the
  `workflow.execute_activity` call, or any Activity-execution code path.

The two must never be conflated. A `RetryPolicy(maximum_attempts=...)` in a workflow is
correct and unrelated; the linter must not flag it, and `max_retries` must never be
evaluated against the Activity execution layer.

### 2. `pydantic_data_converter` on every client and test env  ·  *mechanical*

Recipes serialize Pydantic models as workflow/activity payloads. The converter must be set
on `Client.connect()` (worker + starter) **and** the test `WorkflowEnvironment`, or
serialization diverges between runtime and tests.

```python
client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)  # correct
env = await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)  # correct (tests)
client = await Client.connect("localhost:7233")  # incorrect, Pydantic payloads won't round-trip
```

### 3. Every `execute_activity` sets `start_to_close_timeout`  ·  *mechanical*

An activity with no timeout can hang forever. Always bound it (30s default; raise for
long LLM/research calls).

```python
await workflow.execute_activity(classify, req, start_to_close_timeout=timedelta(seconds=30))  # correct
await workflow.execute_activity(classify, req)  # incorrect, no timeout
```

### 4. Boundary errors raise `ApplicationError(..., non_retryable=True)`  ·  *mechanical (best-effort) + judgment*

In an Activity, catch known-permanent API errors (auth, bad request) and re-raise as a
non-retryable `ApplicationError` so Temporal stops retrying a call that can never succeed.
Transient errors stay retryable (let them propagate).

```python
except anthropic.AuthenticationError as exc:
    raise ApplicationError(str(exc), type="AuthenticationError", non_retryable=True) from exc
```

`recipe-lint` flags an activity that calls an LLM/HTTP client but never raises a
non-retryable `ApplicationError` (best-effort). Whether the *right* errors are classified
is judgment, the reviewer agent checks that.

### 5. Workflows are pure orchestration  ·  *mechanical (best-effort) + judgment*

No I/O, LLM calls, or nondeterminism in `workflows/`. All side effects go through
activities. `recipe-lint` flags LLM/HTTP client imports or construction inside
`workflows/`; subtle nondeterminism (clocks, randomness, ordering) is judgment.

## Naming and project rules

### 6. Naming  ·  *mechanical*  (see [layout.md](layout.md))

Task queue `{recipe-name}-task-queue`; package `cookbook-{recipe-name}-python`; workflow
class `PascalCase`; activity functions `snake_case`.

### 7. Dependency pins  ·  *mechanical*

`pyproject.toml` must declare `requires-python = ">=3.10"` and
`temporalio>=1.15.0,<2`. Do not add an upper Python cap: recipes are runnable apps, and a
cap needlessly excludes newer Python releases.

### 8. Current model names  ·  *mechanical*

Model-name string literals must be current (e.g. `claude-sonnet-4-6`, not a retired
name). `recipe-lint` flags literals matching a known-stale set; the current set lives in
the linter and is easy to update.

## Testing rules  (see also [layout.md](layout.md): tests are mandatory)

### 9. Test markers and skipping env  ·  *mechanical*

Async workflow/activity tests use `@pytest.mark.asyncio` and `@pytest.mark.timeout(30)`,
and run against `WorkflowEnvironment.start_time_skipping(...)` (with the Pydantic
converter, per rule 2).

### 10. Tests mock external calls: no real API key  ·  *judgment*

The suite must pass with no credentials. Activities that hit an LLM/API are mocked (e.g.
`patch(...)` the client, or register a mock activity in the test Worker). That the mocking
genuinely avoids network calls is judgment, the reviewer agent verifies it.

## Quality bar  (from the `python` skill)

### 11. `ruff` clean  ·  *mechanical*

`ruff check` and `ruff format --check` pass. (Step 10 runs ruff via `recipe-lint`.)

### 12. Strict type checking  ·  *mechanical*

`mypy --strict` (or equivalent) passes, type hints on functions, no implicit `Any`.

### 13. Modern, readable Python  ·  *judgment*

Modern idioms (`X | None`, `list[str]`, dataclasses/Pydantic for contracts), clear names,
no dead code. Judgment, the reviewer agent flags smells `ruff`/`mypy` miss.

## Summary: what recipe-lint implements vs what the agent reviews

| # | Rule | Check |
| :-- | :--- | :--- |
| 1 | `max_retries=0` | mechanical |
| 2 | `pydantic_data_converter` (client + test env) | mechanical |
| 3 | `start_to_close_timeout` present | mechanical |
| 4 | non-retryable `ApplicationError` at boundaries | mechanical (best-effort) + judgment |
| 5 | workflows pure orchestration | mechanical (best-effort) + judgment |
| 6 | naming (task queue / package / class / activity) | mechanical |
| 7 | dependency pins | mechanical |
| 8 | current model names | mechanical |
| 9 | test markers + time-skipping env | mechanical |
| 10 | tests mock external calls (no API key) | judgment |
| 11 | `ruff` clean | mechanical |
| 12 | strict typing | mechanical |
| 13 | modern, readable Python | judgment |
