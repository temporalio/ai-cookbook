<!-- 
description: Use the Claim Check pattern to keep large payloads out of Temporal Event History by storing them in Redis and referencing them with keys, with optional codec server support for a better Web UI experience.
tags:[foundations, claim-check, python, redis]
priority: 999
-->

# Claim Check Pattern with Temporal

This recipe demonstrates how to use the Claim Check pattern to offload data from Temporal Server's Event History to external storage. This can be useful in conversational AI applications that include the full conversation history with each LLM call, creating large Event History that can exceed server size limits.

This recipe includes:

- A `PayloadCodec` that stores large payloads in Redis and replaces them with keys
- A client plugin that wires the codec into the Temporal data converter
- A lightweight codec server for a better Web UI experience
- An AI/RAG example workflow that demonstrates the pattern end-to-end

## How the Claim Check Pattern Works

Each Temporal Workflow has an associated Event History that is stored in Temporal Server and used to provide durable execution. When using the Claim Check pattern, we store the payload content of the Event in separate storage system, then store a reference to that storage in the Temporal Event History instead.

The Claim Check Recipe implements a `PayloadCodec` that:

1. Encode: Replaces large payloads with unique keys and stores the original data in external storage (Redis, S3, etc.)
2. Decode: Retrieves the original payload using the key when needed

Workflows operate with small, lightweight keys while maintaining transparent access to full data through automatic encoding/decoding.

## Claim Check Codec Implementation

The `ClaimCheckCodec` implements `PayloadCodec` and adds an inline threshold to keep small payloads inline for debuggability.

*File: claim_check_codec.py*

```python
class ClaimCheckCodec(PayloadCodec):
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, max_inline_bytes: int = 20 * 1024):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.max_inline_bytes = max_inline_bytes

    async def encode(self, payloads: Iterable[Payload]) -> List[Payload]:
        out: List[Payload] = []
        for payload in payloads:
            if len(payload.data or b"") <= self.max_inline_bytes:
                out.append(payload)
                continue
            out.append(await self.encode_payload(payload))
        return out

    async def decode(self, payloads: Iterable[Payload]) -> List[Payload]:
        out: List[Payload] = []
        for payload in payloads:
            if payload.metadata.get("temporal.io/claim-check-codec", b"").decode() != "v1":
                out.append(payload)
                continue
            redis_key = payload.data.decode("utf-8")
            stored_data = await self.redis_client.get(redis_key)
            if stored_data is None:
                raise ValueError(f"Claim check key not found in Redis: {redis_key}")
            out.append(Payload.FromString(stored_data))
        return out
```

### Inline payload threshold

- Default: 20KB
- Where configured: `ClaimCheckCodec(max_inline_bytes=20 * 1024)` in `claim_check_codec.py`
- Change by passing a different `max_inline_bytes` when constructing `ClaimCheckCodec`

## Claim Check Plugin

The `ClaimCheckPlugin` integrates the codec with the Temporal client configuration and supports plugin chaining.

*File: claim_check_plugin.py*

```python
class ClaimCheckPlugin(Plugin):
    def __init__(self):
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self._next_plugin = None

    def init_client_plugin(self, next_plugin: Plugin) -> None:
        self._next_plugin = next_plugin

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        default_converter_class = config["data_converter"].payload_converter_class
        claim_check_codec = ClaimCheckCodec(self.redis_host, self.redis_port)
        config["data_converter"] = DataConverter(
            payload_converter_class=default_converter_class,
            payload_codec=claim_check_codec,
        )
        return self._next_plugin.configure_client(config) if self._next_plugin else config
```

## Example: AI / RAG Workflow using Claim Check

This example ingests a large text, performs lightweight lexical retrieval, and answers a question with an LLM. Large intermediates (chunks, scores) are kept out of Temporal payloads via the Claim Check codec. Only the small final answer is returned inline.

### Activities

*File: activities/ai_claim_check.py*

```python
@activity.defn
async def ingest_document(req: IngestRequest) -> IngestResult:
    text = req.document_bytes.decode("utf-8", errors="ignore")
    chunks = _split_text(text, req.chunk_size, req.chunk_overlap)
    return IngestResult(chunk_texts=chunks, metadata={"filename": req.filename, "mime_type": req.mime_type, "chunk_count": len(chunks)})

@activity.defn
async def rag_answer(req: RagRequest, ingest_result: IngestResult) -> RagAnswer:
    tokenized_corpus = [chunk.split() for chunk in ingest_result.chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = req.question.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(1, req.top_k)]
    contexts = [ingest_result.chunk_texts[i] for i in top_indices]
    chat = await AsyncOpenAI(max_retries=0).chat.completions.create(
        model=req.generation_model,
        messages=[{"role": "user", "content": "..."}],
        temperature=0.2,
    )
    return RagAnswer(answer=chat.choices[0].message.content.strip(), sources=[{"chunk_index": i, "score": float(scores[i])} for i in top_indices])
```

### Workflow

*File: workflows/ai_rag_workflow.py*

```python
@workflow.defn
class AiRagWorkflow:
    @workflow.run
    async def run(self, document_bytes: bytes, filename: str, mime_type: str, question: str) -> RagAnswer:
        ingest = await workflow.execute_activity(
            ingest_document,
            IngestRequest(document_bytes=document_bytes, filename=filename, mime_type=mime_type),
            start_to_close_timeout=timedelta(minutes=10),
            summary="Ingest and embed large document",
        )
        answer = await workflow.execute_activity(
            rag_answer,
            args=[RagRequest(question=question), ingest],
            start_to_close_timeout=timedelta(minutes=5),
            summary="RAG answer using embedded chunks",
        )
        return answer
```

## Configuration

Set environment variables to configure Redis and OpenAI:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export OPENAI_API_KEY=your_key_here
```

## Prerequisites

- Redis server
- Temporal dev server
- Python 3.9+

## Running

1. Start Redis:
```bash
redis-server
```

2. Start Temporal dev server:
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

### Toggle Claim Check (optional)

To demonstrate payload size failures without claim check, you can disable it in your local wiring (e.g., omit the plugin/codec) and re-run. With claim check disabled, large payloads may exceed Temporal's default payload size limits and fail.

## Codec Server for Web UI

When claim check is enabled, the Web UI would otherwise show opaque keys. This codec server shows helpful text with a link to view the raw data on demand.

### Running the Codec Server

```bash
uv run python -m codec_server
```

Then configure the Web UI to use the codec server. For `temporal server start-dev`, see the Temporal docs on configuring codec servers.

### What it shows

Instead of raw keys:

```
abc123-def4-5678-9abc-def012345678
```

You will see text like:

```
"Claim check data (key: abc123-def4-5678-9abc-def012345678) - View at: http://localhost:8081/view/abc123-def4-5678-9abc-def012345678"
```

### Endpoints

- `POST /decode`: Returns helpful text with Redis key and view URL (no data reads)
- `GET /view/{key}`: Serves raw payload data for inspection

The server also includes CORS handling for the local Web UI.
