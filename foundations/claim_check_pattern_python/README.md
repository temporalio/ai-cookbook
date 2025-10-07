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
- `activities/`: Activities that work with large data
- `workflows/`: Workflows that demonstrate the pattern
- `worker.py`: Temporal worker with claim check plugin
- `start_workflow.py`: Example workflow execution
