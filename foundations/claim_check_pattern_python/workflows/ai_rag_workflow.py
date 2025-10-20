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


