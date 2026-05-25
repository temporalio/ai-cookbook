# ADR 002: Use from_callable_with_api_option for Tool Generation

## Status

Accepted

## Context

The original implementation used `FunctionDeclaration.from_callable(client=client, callable=fn)` to generate tool definitions from Python callables. This approach had several drawbacks:

1. **Client dependency**: Required creating a `genai.Client` instance just to generate tool definitions
2. **API key requirement**: The client needed a `GOOGLE_API_KEY`, meaning tool generation couldn't happen without credentials
3. **Sandbox incompatibility**: `genai.Client` uses `threading.local` internally, which is restricted in Temporal's workflow sandbox
4. **Caching workaround**: Required a `_tools_cache` module-level variable and startup-order dependency (worker had to call `get_tools()` before importing the workflow)
5. **Startup coupling**: The worker's `__main__` block had to explicitly pre-populate the cache before the event loop started

## Decision

Adopt `FunctionDeclaration.from_callable_with_api_option(callable=fn, api_option="GEMINI_API")` which accepts the API backend as a string parameter instead of requiring a client instance.

## Consequences

### Positive

- `get_tools()` becomes a pure function with no side effects or external dependencies
- No client or API key needed for tool generation
- Eliminates the `_tools_cache` workaround entirely
- Simplifies worker startup (no pre-initialization step)
- Removes sandbox concerns for tool generation (no `threading.local` involvement)
- Adding new tools is simpler - just add the `from_callable_with_api_option` call

### Negative

- Depends on a newer API that may not be available in older versions of the Google GenAI SDK
