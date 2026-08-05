"""Tests for activities/ai_claim_check.py — _split_text and ingest_document."""

import pytest
from temporalio.testing import ActivityEnvironment

from activities.ai_claim_check import _split_text, ingest_document
from shared.models import IngestRequest


class TestSplitText:
    def test_empty_string(self):
        assert _split_text("", 100, 10) == []

    def test_shorter_than_chunk(self):
        chunks = _split_text("hello world", 100, 10)
        assert chunks == ["hello world"]

    def test_exact_chunk_size(self):
        text = "a" * 100
        chunks = _split_text(text, 100, 10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_chunks_with_overlap(self):
        text = "a" * 250
        chunks = _split_text(text, 100, 20)
        assert len(chunks) > 1
        # Second chunk should start at 80 (100 - 20 overlap)
        assert chunks[1][:20] == chunks[0][80:100]

    def test_all_text_covered(self):
        text = "abcdefghij" * 10  # 100 chars
        chunks = _split_text(text, 30, 5)
        # Reconstruct: each chunk overlaps the previous by 5 chars
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            # Find the overlap
            reconstructed += chunk[5:]  # skip overlap portion
        # All original text should be present
        assert text[0] in reconstructed
        assert text[-1] in reconstructed


class TestIngestDocument:
    @pytest.mark.asyncio
    async def test_text_plain(self):
        env = ActivityEnvironment()
        req = IngestRequest(
            document_bytes=b"Hello world. This is a test document.",
            filename="test.txt",
            mime_type="text/plain",
            chunk_size=20,
            chunk_overlap=5,
        )
        result = await env.run(ingest_document, req)
        assert len(result.chunk_texts) > 0
        assert result.metadata["filename"] == "test.txt"
        assert result.metadata["mime_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_unsupported_mime_type(self):
        env = ActivityEnvironment()
        req = IngestRequest(
            document_bytes=b"data",
            filename="test.pdf",
            mime_type="application/pdf",
        )
        with pytest.raises(ValueError, match="Unsupported MIME type"):
            await env.run(ingest_document, req)
