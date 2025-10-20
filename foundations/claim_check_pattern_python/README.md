# Claim Check Pattern with Temporal

This recipe demonstrates how to use the Claim Check pattern to offload data from Temporal Server's Event History to external storage. This can be useful in conversational AI applications that include the full conversation history with each LLM call, creating large Event History that can exceed server size limits.

## What is the Claim Check Pattern?

Each Temporal Workflow has an associated Event History that is stored in Temporal Server and used to provide durable execution. When using the Claim Check pattern, we store the payload content of the Event in separate storage system, then store a reference to that storage in the Temporal Event History instead.

That is, we:

1. Store large payloads in external storage (Redis, S3, etc.)
2. Replace the payload with a unique key
3. Automatically retrieve the original payload when needed

This is implemented as a `PayloadCodec` that operates transparently - your workflows don't need to know about the claim check mechanism.

## Prerequisites

- **Redis Server**: Required for external storage of large payloads
- **Temporal Server**: Required for workflow execution
- **Python 3.9+**: Required for running the code

## Running the Example

1. Start Redis server:
```bash
redis-server
```

2. Start the Temporal Dev Server:
```bash
temporal server start-dev
```

3. Run the worker:
```bash
uv run python -m worker
```

4. Start execution:
```bash
uv run python -m start_workflow
```

## Configuration

The example uses Redis for external storage. You can configure the Redis connection with environment variables:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### Inline payload threshold (skip claim check for small payloads)

By default, payloads that are small enough are kept inline to improve debuggability and avoid unnecessary indirection. This example sets the inline threshold to 20KB. Any payload larger than 20KB will be claim-checked and stored in Redis; payloads at or below 20KB remain inline.

- Default: 20KB
- Where configured: `ClaimCheckCodec(max_inline_bytes=20 * 1024)` in `claim_check_codec.py`
- How to change: pass a different `max_inline_bytes` when constructing `ClaimCheckCodec` (e.g., in your client/plugin wiring)

## Key Components

- `claim_check_codec.py`: Implements the PayloadCodec for claim check functionality
- `claim_check_plugin.py`: Temporal plugin that integrates the codec
- `codec_server.py`: Lightweight codec server for Web UI integration
- `activities/`: Activities that demonstrate large data processing:
  - `transform_large_dataset`: Transforms large input into large output
  - `generate_summary`: Takes large input and produces small summary
- `workflows/`: Workflows that demonstrate the pattern
- `worker.py`: Temporal worker with claim check plugin
- `start_workflow.py`: Example workflow execution

## AI / RAG Example using Claim Check

This example also includes a simple Retrieval-Augmented Generation (RAG) flow that ingests a large text (a public-domain book), creates embeddings, and answers a question while keeping large intermediates (chunks, embeddings) out of Temporal payloads via the Claim Check codec. Only the small final answer is returned inline.

### Files

- `activities/ai_claim_check.py`: Activities `ingest_document` and `rag_answer` using OpenAI.
- `workflows/ai_rag_workflow.py`: Orchestrates ingestion then question answering.
- `start_workflow.py`: Starter that downloads a public-domain text if missing and asks a question.

### Requirements

- Set `OPENAI_API_KEY` for embeddings and chat generation.
- Redis and Temporal dev server running (same as the main example).
- Internet access for the first run to download the text from Project Gutenberg (`https://www.gutenberg.org/ebooks/100.txt.utf-8`).

### Run

1. Export your API key:
```bash
export OPENAI_API_KEY=your_key_here
```
2. Start the worker (claim check enabled by default):
```bash
uv run python -m worker
```
3. Start the AI/RAG workflow (first run will download the text):
```bash
uv run python -m start_workflow
```

### Toggle Claim Check (optional)

To demonstrate payload size failures without claim check, disable it with an environment variable:

```bash
export CLAIM_CHECK_ENABLED=false
uv run python -m worker
uv run python -m start_workflow
```

With claim check disabled, large payloads (e.g., the Shakespeare text or large intermediates) may exceed Temporal's default payload size limits and fail. Re-enable by unsetting or setting `CLAIM_CHECK_ENABLED=true`.

The starter downloads “The Complete Works of William Shakespeare” from Project Gutenberg [link](https://www.gutenberg.org/ebooks/100.txt.utf-8) on first run and saves it under `assets/shakespeare_complete.txt` (~5.1MB). This exceeds Temporal’s default payload size (2MB), making it a good demonstration for the claim check pattern. Large intermediates (chunked text and embeddings) will be claim-checked automatically (payloads > 20KB stored in Redis). The final `RagAnswer` is small and remains inline for easy inspection in the Web UI.

## How It Works

This example demonstrates the claim check pattern with a realistic data processing pipeline:

1. **Large Workflow Input**: The workflow receives a large dataset from the client
2. **Large Activity Input/Output**: The first activity transforms the large dataset, producing another large dataset
3. **Large Activity Input, Small Output**: The second activity takes the transformed data and produces a compact summary

This flow shows how the claim check pattern handles large payloads at multiple stages of processing, making it transparent to your workflow logic while avoiding Temporal's payload size limits.

## Codec Server for Web UI

When using the Claim Check pattern, the Temporal Web UI will show encoded Redis keys instead of the actual payload data. This makes debugging and monitoring difficult since you can't see what data is being passed through your workflows.

### The Problem

Without a codec server, the Web UI displays raw claim check keys like:
```
abc123-def4-5678-9abc-def012345678
```

This provides no context about what data is stored or how to access it, making workflow debugging and monitoring challenging.

### Our Solution: Lightweight Codec Server

The codec server provides helpful information without reading large payload data during Web UI operations.

Instead of raw keys, the Web UI displays:
```
"Claim check data (key: abc123-def4-5678-9abc-def012345678) - View at: http://localhost:8081/view/abc123-def4-5678-9abc-def012345678"
```

This gives you the Redis key and a direct link to view the full payload data when needed.

### Running the Codec Server

1. Start the codec server:
```bash
uv run python -m codec_server
```

2. Configure the Temporal Web UI to use the codec server. For `temporal server start-dev`, see the [Temporal documentation on configuring codec servers](https://docs.temporal.io/production-deployment/data-encryption#set-your-codec-server-endpoints-with-web-ui-and-cli) for the appropriate configuration method.

3. Access the Temporal Web UI and you'll see helpful summaries instead of raw keys.

### Configuration Details

The codec server implements the Temporal codec server protocol with two endpoints:

- **`/decode`**: Returns helpful text with Redis key and view URL
- **`/view/{key}`**: Serves the raw payload data for inspection

When you click the view URL, you'll see the complete payload data as stored in Redis, formatted appropriately for text or binary content.
