from typing import List

from temporalio import activity

from shared.models import IngestRequest, IngestResult, RagRequest, RagAnswer


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
    # Import heavy dependencies inside the function, not at module level
    # This prevents NumPy from being loaded into the workflow sandbox
    from openai import AsyncOpenAI
    from rank_bm25 import BM25Okapi
    
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


