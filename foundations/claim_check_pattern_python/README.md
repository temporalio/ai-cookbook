# Claim Check Pattern with Temporal

This recipe demonstrates how to use the Claim Check pattern to efficiently handle large payloads in Temporal workflows by storing them externally and passing only keys through the system.

## What is the Claim Check Pattern?

The Claim Check pattern allows you to work with large data in Temporal workflows without hitting payload size limits or performance issues. Instead of passing large payloads directly through Temporal, the pattern:

1. Stores large payloads in external storage (Redis, S3, etc.)
2. Replaces the payload with a unique key
3. Automatically retrieves the original payload when needed

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
