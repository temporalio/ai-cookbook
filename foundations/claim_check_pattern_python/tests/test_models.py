"""Tests for shared/models.py — dataclass construction and defaults."""

from shared.models import IngestRequest, IngestResult, RagRequest, RagAnswer


class TestIngestRequest:
    def test_defaults(self):
        req = IngestRequest(
            document_bytes=b"hello",
            filename="test.txt",
            mime_type="text/plain",
        )
        assert req.chunk_size == 1500
        assert req.chunk_overlap == 200
        assert req.embedding_model == "text-embedding-3-large"

    def test_custom_values(self):
        req = IngestRequest(
            document_bytes=b"data",
            filename="doc.txt",
            mime_type="text/plain",
            chunk_size=500,
            chunk_overlap=50,
        )
        assert req.chunk_size == 500
        assert req.chunk_overlap == 50


class TestIngestResult:
    def test_construction(self):
        result = IngestResult(
            chunk_texts=["chunk1", "chunk2"],
            metadata={"filename": "test.txt", "chunk_count": 2},
        )
        assert len(result.chunk_texts) == 2
        assert result.metadata["chunk_count"] == 2


class TestRagRequest:
    def test_defaults(self):
        req = RagRequest(question="What is life?")
        assert req.top_k == 4
        assert req.generation_model == "gpt-4o-mini"


class TestRagAnswer:
    def test_construction(self):
        answer = RagAnswer(
            answer="42",
            sources=[{"chunk_index": 0, "score": 1.0}],
        )
        assert answer.answer == "42"
        assert len(answer.sources) == 1
