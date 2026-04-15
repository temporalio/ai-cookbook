# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Temporal AI Cookbook** - a monorepo of example implementations showing how to build durable AI agents and LLM-powered applications using Temporal. Each subdirectory is an independent project demonstrating specific patterns.

## Repository Structure

```
agents/                     # Advanced agent implementations
  agentic_loop_tool_call_openai_python/   # OpenAI with agentic loop
  agentic_loop_tool_call_claude_python/   # Claude with agentic loop
  human_in_the_loop_python/               # Approval workflows via Signals
  openai_agents_sdk_python/               # OpenAI Agents SDK integration
  google_genai_sdk_python/                # Google Gemini implementation

foundations/                # Basic building blocks
  hello_world_openai_responses_python/    # Minimal LLM invocation
  tool_call_openai_python/                # Single-turn tool calling
  structured_output_openai_responses_python/
  claim_check_pattern_python/

deep_research/              # Multi-agent research patterns
mcp/                        # Model Context Protocol examples
```

## Development Commands

Each project is independent. Work within a specific project directory:

```bash
cd agents/agentic_loop_tool_call_openai_python

# Install dependencies
uv sync

# Start Temporal Dev Server (in separate terminal)
temporal server start-dev

# Start worker
uv run python -m worker

# Run example (in another terminal)
uv run python -m start_workflow "your prompt here"
```

## Environment Variables

Set API keys before running workers:
- `OPENAI_API_KEY` - for OpenAI-based examples
- `ANTHROPIC_API_KEY` - for Claude-based examples
- `GOOGLE_API_KEY` - for Google Gemini examples

## Key Architecture Patterns

**Workflow → Activity → LLM/Tool pattern**: Workflows orchestrate, activities wrap external calls (LLM APIs, tools). This provides durability and automatic retries.

**Disable client retries**: All examples set `max_retries=0` on LLM clients because Temporal handles retries:
```python
client = AsyncOpenAI(max_retries=0)
```

**Dynamic activities for tools**: Tools are loosely coupled from agent workflows using dynamic activities, allowing tools to be added/removed without changing agent code.

**Pydantic data converter**: Required for serializing complex types:
```python
from temporalio.contrib.pydantic import pydantic_data_converter
client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
```

**Human-in-the-loop**: Use Temporal Signals for injecting human decisions into waiting workflows. Workflows can wait indefinitely without consuming compute.

## CI/CD

The `docs-build-check.yml` workflow validates that changes don't break the Temporal documentation site build. It runs on PRs to main.

## README Format

Each project README uses HTML comments for documentation metadata:
```markdown
<!--
description: Brief description
tags: [agents, python]
priority: 750
-->
```
