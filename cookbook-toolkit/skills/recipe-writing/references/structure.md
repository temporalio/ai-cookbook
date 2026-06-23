# Recipe README structure

The README **is** the recipe's documentation. The docs site (`bin/sync-ai-cookbook.js`
in `temporalio/documentation`) publishes each recipe's `README.md` verbatim as its page,
so the README's structure is user-facing, not just repo metadata.

The canonical recipe README is a **light code walkthrough**: show each file's code in
build order, with a sentence or two of *why* around it, ending with how to run it. A
recipe is "here's the code, here's how it works", not a tutorial, a Validated Pattern,
or a course.

## The canonical shape

In order:

1. `# Title`: the mandatory H1 (see below).
2. **Intro**: 1-2 sentences: what the recipe demonstrates and the one framing line that
   orients an AI engineer (often "the external call happens in a Temporal Activity").
3. **Optional orientation**: either a short "key design decisions" bullet list *or* a
   `## Application Components` overview. Use it when the recipe has several moving parts;
   skip it for trivial ones.
4. **A sequence of `## Create the {Component}` sections**: one per file/component, in the
   order you'd build them: typically Activity → Workflow → Worker → Workflow Starter, plus
   tools / agents / helpers as the recipe needs.
5. `## Running`: start the dev server, the worker, and the starter.

Skeleton:

````markdown
# Title

One or two sentences on what this demonstrates and where the LLM call lives.

## Create the Activity
...introduce → *File: ...* → code → explain...

## Create the Workflow
...

## Create the Worker
...

## Create the Workflow Starter
...

## Running
````

## Mandatory H1

Every README must begin with a single `# Title`. The docs sync promotes the H1 to the
published page title and **hard-fails the docs build if it is missing** (`missing H1
title`). Make the title the recipe's name as it should appear in the cookbook,
outcome/action oriented, not "How to X with Y on Z."

## The light code-sandwich

Each `## Create the {Component}` section follows **introduce → show → explain**:

1. **Introduce**: one or two sentences on what the file does and the design decision it
   embodies.
2. **Show**: a `*File: path*` line naming the file, immediately followed by the fenced
   code block.
3. **Explain**: call out the key lines or gotchas a reader needs (e.g. *why*
   `max_retries=0`).

Keep it light. No Executive Summary / Problem Statement / Outcomes scaffolding, that is
Validated Patterns, not the cookbook. The code is the hero; prose is the connective
tissue between blocks.

### Example (from `foundations/hello_world_openai_responses_python`)

````markdown
## Create the Activity

We create a wrapper for the `create` method of the `AsyncOpenAI` client object.
This is a generic Activity that invokes the OpenAI LLM.

We set `max_retries=0` when creating the `AsyncOpenAI` client.
This moves the responsibility for retries from the OpenAI client to Temporal.

*File: activities/openai_responses.py*

```python
@activity.defn
async def create(request: OpenAIResponsesRequest) -> Response:
    # Temporal best practice: Disable retry logic in the OpenAI client library.
    client = AsyncOpenAI(max_retries=0)
    resp = await client.responses.create(
        model=request.model,
        instructions=request.instructions,
        input=request.input,
        timeout=15,
    )
    return resp
```
````

The intro states what the Activity is and why; the `*File:*` line anchors the code to a
real path; the explanation surfaces the one decision that matters (`max_retries=0`).

## Allowed variation

- **Section names track the recipe.** `## Create the Activity for LLM invocations`,
  `## Create the Agent`, and `## Initiate an interaction with the agent` are all fine. The
  pattern is `## Create the {thing}` in build order, followed by a run section.
- **Trivial single-file recipes may collapse sections**, but must keep: the H1, a 1-2
  sentence intro, and a `## Running` (or "Run it") section.
- **Diagrams/images** live in `_assets/` and are referenced inline where they aid
  understanding; pair an architecture diagram with a short numbered walk-through.

## Non-canonical (do not use)

The brief "overview + prerequisites + run + expected output" style, with no inline code,
is **not** canonical. It renders as a thin docs page that never shows the reader the code
the recipe is supposed to teach. Recipes currently written this way are Phase-3 backfill
targets, not models to copy.

## Verified against the corpus

The canonical shape is confirmed against `foundations/hello_world_openai_responses_python`,
`foundations/hello_world_litellm_python`, and `agents/tool_call_openai_python`: all use
`## Create the {Component}` sections in build order ending in `## Running`.

Some recipes deviate from this shape:

- `agents/human_in_the_loop_python` uses a richer Overview / Setup / Architecture / Key
  Patterns shape, a sanctioned variant.
- Larger recipes (`deep_research/basic_openai_python`, `foundations/claim_check_pattern_python`,
  `mcp/hello_world_durable_mcp_server`) use the walkthrough shape with extra sections,
  acceptable as long as the H1 → Create-the-X → Running spine holds.
