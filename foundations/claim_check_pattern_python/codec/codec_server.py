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