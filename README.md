# Temporal AI Cookbook

A collection of example projects and patterns for building durable AI agents and workflows with [Temporal](https://temporal.io/). Each recipe is a self-contained, runnable project that demonstrates a specific concept or integration pattern.

## Examples

| Category | Description |
|---|---|
| [`foundations/`](foundations/) | Core building blocks: hello world, structured output, retry patterns, claim-check |
| [`agents/`](agents/) | Agentic loops, tool calling, human-in-the-loop approval flows |
| [`deep_research/`](deep_research/) | Multi-agent research systems with planning, web search, and synthesis |
| [`mcp/`](mcp/) | Durable [Model Context Protocol](https://modelcontextprotocol.io/) servers backed by Temporal workflows |


## Toolkit

This repository serves three purposes:

1. **Recipe content** — the runnable examples above. Each recipe's `README.md` is published
   to [docs.temporal.io/ai-cookbook](https://docs.temporal.io/ai-cookbook).
2. **Consistency tooling** — `cookbook-toolkit/tools/recipe-lint`, a `uv`-runnable CLI that checks a
   recipe's structure, layout, naming, and Temporal/Python conventions, plus a small
   [Vale](https://vale.sh/) prose ruleset under `cookbook-toolkit/styles/`. CI runs both advisorily
   (see `.github/workflows/lint-recipes.yml`); only docs-breakers and missing tests fail.
3. **A local Claude Code plugin** — `cookbook-toolkit/` is a Claude Code plugin (the `recipe-writing`
   skill, the `recipe-reviewer` agent, and the `/recipe-ify`, `/recipe-scout`,
   `/new-recipe`, and `/review-recipe` commands). It is loaded locally, not published:

   ```bash
   claude --plugin-dir <path-to-repo>/cookbook-toolkit
   ```

The recipe conventions live in exactly one place —
`cookbook-toolkit/skills/recipe-writing/references/` — which the linter, commands, and agent all
reference. Run the linter directly without the plugin:

```bash
uv run --project cookbook-toolkit/tools/recipe-lint recipe-lint <category>/<recipe>_python
```

(Local `.claude/` skills remain fine for one-off use; `cookbook-toolkit/` is the durable home.)

## Learn More

- **Documentation:** [docs.temporal.io](https://docs.temporal.io)
- **AI Cookbook site:** [docs.temporal.io/ai-cookbook](https://docs.temporal.io/ai-cookbook)
- **Community Slack:** [temporalio.slack.com](https://temporalio.slack.com)
- **Community Forum:** [community.temporal.io](https://community.temporal.io)
