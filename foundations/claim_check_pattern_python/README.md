<!-- 
description: Use the Claim Check pattern to handle large payloads to workflows and activities.
tags:[foundations, claim-check, python, s3]
priority: 400
-->

# Claim Check Pattern with Temporal

This recipe demonstrates how to use the Claim Check pattern to offload data from Temporal Server's Event History to external storage. This can be useful in conversational AI applications that include the full conversation history with each LLM call, creating large Event History that can exceed server size limits.

This recipe includes:

- A `PayloadCodec` ([docs](https://docs.temporal.io/payload-codec)) that stores large payloads in S3 and replaces them with keys
- A client [plugin](https://docs.temporal.io/develop/plugins-guide) that wires the codec into the Temporal data converter
- A lightweight codec server for a better Web UI experience
- An AI/RAG example workflow that demonstrates the pattern end-to-end

## How the Claim Check Pattern Works

Each Temporal Workflow has an associated Event History that is stored in Temporal Server and used to provide durable execution. When using the Claim Check pattern, we store the payload content of the Event in separate storage system, then store a reference to that storage in the Temporal Event History instead.

The Claim Check Recipe implements a `PayloadCodec` that:

1. Encode: Replaces large payloads with unique keys and stores the original data in external storage (S3, Database, etc.)
2. Decode: Retrieves the original payload using the key when needed

Workflows operate with small, lightweight keys while maintaining transparent access to full data through automatic encoding/decoding.

## Claim Check Codec Implementation

The `ClaimCheckCodec` implements `PayloadCodec` and adds an inline threshold to keep small payloads inline. This avoids the latency costs of uploading/downloading the payload externally when it's not required.

*File: codec/claim_check.py*

```python
import uuid
from typing import Iterable, List
import aioboto3
from botocore.exceptions import ClientError

from temporalio.api.common.v1 import Payload
from temporalio.converter import PayloadCodec


class ClaimCheckCodec(PayloadCodec):
    """PayloadCodec that implements the Claim Check pattern using S3 storage.
    
    This codec stores large payloads in S3 and replaces them with unique keys,
    allowing Temporal workflows to operate with lightweight references instead
    of large payload data.
    """

    def __init__(self, 
                 bucket_name: str = "temporal-claim-check",
                 endpoint_url: str = None,
                 region_name: str = "us-east-1",
                 max_inline_bytes: int = 20 * 1024):
        """Initialize the claim check codec with S3 connection details.
        
        Args:
            bucket_name: S3 bucket name for storing claim check data
            endpoint_url: S3 endpoint URL (for MinIO or other S3-compatible services)
            region_name: AWS region name
            max_inline_bytes: Payloads up to this size will be left inline
        """
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.max_inline_bytes = max_inline_bytes
        
        # Initialize aioboto3 session
        self.session = aioboto3.Session()
        
        # Ensure bucket exists
        self._bucket_created = False

    async def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, creating it if necessary."""
        if self._bucket_created:
            return
            
        async with self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            region_name=self.region_name
        ) as s3_client:
            try:
                await s3_client.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code in ['404', 'NoSuchBucket']:
                    try:
                        await s3_client.create_bucket(Bucket=self.bucket_name)
                    except ClientError as create_error:
                        # Handle bucket already exists race condition
                        if create_error.response['Error']['Code'] not in ['BucketAlreadyExists', 'BucketAlreadyOwnedByYou']:
                            raise create_error
                elif error_code not in ['403', 'Forbidden']:
                    raise e
        
        self._bucket_created = True

    async def encode(self, payloads: Iterable[Payload]) -> List[Payload]:
        """Replace large payloads with keys and store original data in S3.
        
        Args:
            payloads: Iterable of payloads to encode
            
        Returns:
            List of encoded payloads (keys for claim-checked payloads)
        """
        await self._ensure_bucket_exists()
        
        out: List[Payload] = []
        for payload in payloads:
            # Leave small payloads inline to improve debuggability and avoid unnecessary indirection
            data_size = len(payload.data or b"")
            if data_size <= self.max_inline_bytes:
                out.append(payload)
                continue

            encoded = await self.encode_payload(payload)
            out.append(encoded)
        return out

    async def decode(self, payloads: Iterable[Payload]) -> List[Payload]:
        """Retrieve original payloads from S3 using stored keys.
        
        Args:
            payloads: Iterable of payloads to decode
            
        Returns:
            List of decoded payloads (original data retrieved from S3)
            
        Raises:
            ValueError: If a claim check key is not found in S3
        """
        await self._ensure_bucket_exists()
        
        out: List[Payload] = []
        for payload in payloads:
            if payload.metadata.get("temporal.io/claim-check-codec", b"").decode() != "v1":
                # Not a claim-checked payload, pass through unchanged
                out.append(payload)
                continue

            s3_key = payload.data.decode("utf-8")
            stored_data = await self.get_payload_from_s3(s3_key)
            if stored_data is None:
                raise ValueError(f"Claim check key not found in S3: {s3_key}")
            
            original_payload = Payload.FromString(stored_data)
            out.append(original_payload)
        return out

    async def encode_payload(self, payload: Payload) -> Payload:
        """Store payload in S3 and return a key-based payload.
        
        Args:
            payload: Original payload to store
            
        Returns:
            Payload containing only the S3 key
        """
        await self._ensure_bucket_exists()
        
        key = str(uuid.uuid4())
        serialized_data = payload.SerializeToString()
        
        # Store the original payload data in S3
        async with self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            region_name=self.region_name
        ) as s3_client:
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=serialized_data
            )
        
        # Return a lightweight payload containing only the key
        return Payload(
            metadata={
                "encoding": b"claim-checked",
                "temporal.io/claim-check-codec": b"v1",
            },
            data=key.encode("utf-8"),
        )

    async def get_payload_from_s3(self, s3_key: str) -> bytes:
        """Retrieve payload data from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Raw payload data bytes, or None if not found
        """
        try:
            async with self.session.client(
                's3',
                endpoint_url=self.endpoint_url,
                region_name=self.region_name
            ) as s3_client:
                response = await s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                return await response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise e
```

### Inline payload threshold

- Default: 20KB
- Where configured: `ClaimCheckCodec(max_inline_bytes=20 * 1024)` in `codec/claim_check.py`
- Change by passing a different `max_inline_bytes` when constructing `ClaimCheckCodec`

## Claim Check Plugin

The `ClaimCheckPlugin` integrates the codec with the Temporal client configuration and supports plugin chaining.

*File: codec/plugin.py*

```python
import os
from temporalio.client import Plugin, ClientConfig
from temporalio.converter import DataConverter
from temporalio.service import ConnectConfig, ServiceClient

from .claim_check import ClaimCheckCodec


class ClaimCheckPlugin(Plugin):
    """Temporal plugin that integrates the Claim Check codec with client configuration."""

    def __init__(self):
        """Initialize the plugin with S3 connection configuration."""
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "temporal-claim-check")
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        self._next_plugin = None

    def init_client_plugin(self, next_plugin: Plugin) -> None:
        """Initialize this plugin in the client plugin chain."""
        self._next_plugin = next_plugin

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        """Apply the claim check configuration to the client.
        
        Args:
            config: Temporal client configuration
            
        Returns:
            Updated client configuration with claim check data converter
        """
        # Configure the data converter with claim check codec
        default_converter_class = config["data_converter"].payload_converter_class
        claim_check_codec = ClaimCheckCodec(
            bucket_name=self.bucket_name,
            endpoint_url=self.endpoint_url,
            region_name=self.region_name
        )
        
        config["data_converter"] = DataConverter(
            payload_converter_class=default_converter_class,
            payload_codec=claim_check_codec
        )
        
        # Delegate to next plugin if it exists
        if self._next_plugin:
            return self._next_plugin.configure_client(config)
        return config

    async def connect_service_client(self, config: ConnectConfig) -> ServiceClient:
        """Connect to the Temporal service.
        
        Args:
            config: Service connection configuration
            
        Returns:
            Connected service client
        """
        # Delegate to next plugin if it exists
        if self._next_plugin:
            return await self._next_plugin.connect_service_client(config)
        
        # If no next plugin, use default connection
        from temporalio.service import ServiceClient
        return await ServiceClient.connect(config)
```

## Example: AI / RAG Workflow using Claim Check

This example ingests a large text, performs lightweight lexical retrieval, and answers a question with an LLM. Large intermediates (chunks, scores) are kept out of Temporal payloads via the Claim Check codec. Only the small final answer is returned inline.

### Activities

*File: activities/ai_claim_check.py*

```python
from dataclasses import dataclass
from typing import List, Dict, Any

from temporalio import activity

from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi


@dataclass
class IngestRequest:
    document_bytes: bytes
    filename: str
    mime_type: str
    chunk_size: int = 1500
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-large"


@dataclass
class IngestResult:
    chunk_texts: List[str]
    metadata: Dict[str, Any]


@dataclass
class RagRequest:
    question: str
    top_k: int = 4
    generation_model: str = "gpt-4o-mini"


@dataclass
class RagAnswer:
    answer: str
    sources: List[Dict[str, Any]]


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


@activity.defn
async def ingest_document(req: IngestRequest) -> IngestResult:
    # Convert bytes to text. For PDFs/audio/images, integrate proper extractors.
    if req.mime_type != "text/plain":
        raise ValueError(f"Unsupported MIME type: {req.mime_type}")

    text = req.document_bytes.decode("utf-8", errors="ignore")
    chunks = _split_text(text, req.chunk_size, req.chunk_overlap)
    return IngestResult(
        chunk_texts=chunks,
        metadata={
            "filename": req.filename,
            "mime_type": req.mime_type,
            "chunk_count": len(chunks),
        },
    )


@activity.defn
async def rag_answer(req: RagRequest, ingest_result: IngestResult) -> RagAnswer:
    client = AsyncOpenAI(max_retries=0)

    # Lexical retrieval using BM25 over chunk texts
    # Simple whitespace tokenization
    tokenized_corpus: List[List[str]] = [chunk.split() for chunk in ingest_result.chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = req.question.split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(1, req.top_k)]
    contexts = [ingest_result.chunk_texts[i] for i in top_indices]
    sources = [{"chunk_index": i, "score": float(scores[i])} for i in top_indices]

    prompt = (
        "Use the provided context chunks to answer the question.\n\n"
        f"Question: {req.question}\n\n"
        "Context:\n" + "\n---\n".join(contexts) + "\n\nAnswer:"
    )

    chat = await client.chat.completions.create(
        model=req.generation_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = chat.choices[0].message.content.strip()

    return RagAnswer(answer=answer, sources=sources)
```

### Workflow

*File: workflows/ai_rag_workflow.py*

```python
from temporalio import workflow
from datetime import timedelta

from activities.ai_claim_check import (
    IngestRequest,
    IngestResult,
    RagRequest,
    RagAnswer,
    ingest_document,
    rag_answer,
)


@workflow.defn
class AiRagWorkflow:
    @workflow.run
    async def run(self, document_bytes: bytes, filename: str, mime_type: str, question: str) -> RagAnswer:
        ingest: IngestResult = await workflow.execute_activity(
            ingest_document,
            IngestRequest(
                document_bytes=document_bytes,
                filename=filename,
                mime_type=mime_type,
            ),
            start_to_close_timeout=timedelta(minutes=10),
            summary="Ingest and embed large document",
        )

        answer: RagAnswer = await workflow.execute_activity(
            rag_answer,
            args=[
                RagRequest(question=question),
                ingest,
            ],
            start_to_close_timeout=timedelta(minutes=5),
            summary="RAG answer using embedded chunks",
        )
        return answer
```

## Running

### Prerequisites

- MinIO server (for local testing) or AWS S3 access (for production)
- Temporal dev server
- Python 3.9+

### Configuration

Set environment variables to configure S3 and OpenAI:

```bash
# For MinIO (recommended for local testing)
export S3_ENDPOINT_URL=http://localhost:9000
export S3_BUCKET_NAME=temporal-claim-check
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1

# For production AWS S3
# export S3_BUCKET_NAME=your-bucket-name
# export AWS_REGION=us-east-1
# you may use access keys or...
# export AWS_ACCESS_KEY_ID=your-access-key
# export AWS_SECRET_ACCESS_KEY=your-secret-key
# ... sso
# export AWS_profile=your-profile
# aws sso login --profile your-profile

export OPENAI_API_KEY=your_key_here
```

### Option 1: MinIO (Recommended for Testing)

1. Start MinIO:
```bash
docker run -d -p 9000:9000 -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  quay.io/minio/minio server /data --console-address ":9001"
```

The bucket will be auto-created by the code. You can view stored objects in the MinIO web console at http://localhost:9001 (credentials: minioadmin/minioadmin).

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

### Option 2: AWS S3 (Production)

1. Create an S3 bucket in your AWS account
2. Configure AWS credentials (via AWS CLI, environment variables, IAM roles or sso)
3. Set the environment variables for your bucket
4. Follow steps 2-4 from Option 1

### Toggle Claim Check (optional)

To demonstrate payload size failures without claim check, you can disable it in your local wiring (e.g., omit the plugin/codec) and re-run. With claim check disabled, large payloads may exceed Temporal's default payload size limits and fail.

## Codec Server for Web UI

When claim check is enabled, the Web UI would otherwise show opaque keys. This codec server shows helpful text with a link to view the raw data on demand.

*File: codec/codec_server.py*

```python
from functools import partial
from typing import Awaitable, Callable, Iterable, List
import json
import os

from aiohttp import hdrs, web
from google.protobuf import json_format
from temporalio.api.common.v1 import Payload, Payloads

from .claim_check import ClaimCheckCodec

def build_codec_server() -> web.Application:
    # Create codec with environment variable configuration (same as plugin)
    codec = ClaimCheckCodec(
        bucket_name=os.getenv("S3_BUCKET_NAME", "temporal-claim-check"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    
    # Configure Web UI endpoint
    temporal_web_url = os.getenv("TEMPORAL_WEB_URL", "http://localhost:8233")
    # Configure codec server endpoint for viewing raw data
    codec_server_url = os.getenv("CODEC_SERVER_URL", "http://localhost:8081")
    
    # CORS handler - needed because Temporal Web UI runs on a different port/domain
    # and the browser blocks cross-origin requests by default; CORS headers allow these requests
    async def cors_options(req: web.Request) -> web.Response:
        resp = web.Response()
        if req.headers.get(hdrs.ORIGIN) == temporal_web_url:
            resp.headers[hdrs.ACCESS_CONTROL_ALLOW_ORIGIN] = temporal_web_url
            resp.headers[hdrs.ACCESS_CONTROL_ALLOW_METHODS] = "POST"
            resp.headers[hdrs.ACCESS_CONTROL_ALLOW_HEADERS] = "content-type,x-namespace"
        return resp

    # Custom decode function that provides URLs to view raw data
    async def decode_with_urls(payloads: Iterable[Payload]) -> List[Payload]:
        """Decode claim check payloads and provide URLs to view the raw data."""
        out: List[Payload] = []
        
        for payload in payloads:
            if payload.metadata.get("temporal.io/claim-check-codec", b"").decode() != "v1":
                # Not a claim-checked payload, pass through unchanged
                out.append(payload)
                continue

            # Get the S3 key
            s3_key = payload.data.decode("utf-8")
            
            # Return simple text with link - no data reading
            link_text = f"Claim check data (key: {s3_key}) - View at: {codec_server_url}/view/{s3_key}"
            
            summary_payload = Payload(
                metadata={"encoding": b"json/plain"},
                data=json.dumps({"text": link_text}).encode("utf-8")
            )
            out.append(summary_payload)
        
        return out

    # Endpoint to view raw payload data
    async def view_raw_data(req: web.Request) -> web.Response:
        """View the raw payload data for a given S3 key."""
        s3_key = req.match_info['key']
        
        try:
            stored_data = await codec.get_payload_from_s3(s3_key)
            if stored_data is None:
                return web.Response(
                    text=json.dumps({"error": f"Key not found: {s3_key}"}),
                    content_type="application/json",
                    status=404
                )
            
            # Parse and return the original payload
            original_payload = Payload.FromString(stored_data)
            
            # Try to decode as text, fall back to base64 for binary data
            try:
                data_text = original_payload.data.decode("utf-8")
                return web.Response(
                    text=data_text,
                    content_type="text/plain"
                )
            except UnicodeDecodeError:
                import base64
                data_b64 = base64.b64encode(original_payload.data).decode("utf-8")
                return web.Response(
                    text=f"Binary data (base64):\n{data_b64}",
                    content_type="text/plain"
                )
                
        except Exception as e:
            return web.Response(
                text=json.dumps({"error": f"Failed to retrieve data: {str(e)}"}),
                content_type="application/json",
                status=500
            )

    # General purpose payloads-to-payloads
    async def apply(
        fn: Callable[[Iterable[Payload]], Awaitable[List[Payload]]], req: web.Request
    ) -> web.Response:
        # Read payloads as JSON
        assert req.content_type == "application/json"
        data = await req.read()
        payloads = json_format.Parse(data.decode("utf-8"), Payloads())
        
        # Apply
        payloads = Payloads(payloads=await fn(payloads.payloads))

        # Apply CORS and return JSON
        resp = await cors_options(req)
        resp.content_type = "application/json"
        resp.text = json_format.MessageToJson(payloads)

        return resp

    # Build app
    app = web.Application()
    app.add_routes(
        [
            web.post("/encode", partial(apply, codec.encode)),
            web.post("/decode", partial(apply, decode_with_urls)),
            web.get("/view/{key}", view_raw_data),
            web.options("/decode", cors_options),
        ]
    )
    return app


if __name__ == "__main__":
    web.run_app(build_codec_server(), host="127.0.0.1", port=8081)
```

### Running the Codec Server

```bash
uv run python -m codec.codec_server
```

Then [configure the Web UI to use the codec server](https://docs.temporal.io/production-deployment/data-encryption#set-your-codec-server-endpoints-with-web-ui-and-cli).

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

- `POST /encode`: Encodes payloads using the claim check codec
- `POST /decode`: Returns helpful text with S3 key and view URL (no data reads)
- `GET /view/{key}`: Serves raw payload data for inspection
- `OPTIONS /decode`: Handles CORS preflight requests

The server also includes CORS handling for the local Web UI.
