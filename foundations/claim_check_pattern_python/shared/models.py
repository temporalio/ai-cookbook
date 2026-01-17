from dataclasses import dataclass
from typing import List, Dict, Any


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
