import asyncio
import time
import os
from pathlib import Path

import requests
from temporalio.client import Client

from claim_check_plugin import ClaimCheckPlugin
from workflows.ai_rag_workflow import AiRagWorkflow


ASSETS_DIR = Path(os.path.join(os.path.dirname(__file__), "assets"))
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

GUTENBERG_URL = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
LOCAL_FILENAME = "shakespeare_complete.txt"
ASSET_PATH = ASSETS_DIR / LOCAL_FILENAME


async def main():
    claim_check_enabled = os.getenv("CLAIM_CHECK_ENABLED", "true").lower() != "false"
    plugins = [ClaimCheckPlugin()] if claim_check_enabled else []

    client = await Client.connect(
        "localhost:7233",
        plugins=plugins,
    )

    if not ASSET_PATH.exists():
        print(f"Downloading public-domain text from {GUTENBERG_URL} ...")
        resp = requests.get(GUTENBERG_URL, timeout=60)
        resp.raise_for_status()
        ASSET_PATH.write_bytes(resp.content)
        print(f"Saved to {ASSET_PATH}")

    doc_bytes = ASSET_PATH.read_bytes()

    result = await client.execute_workflow(
        AiRagWorkflow.run,
        args=[
            doc_bytes,
            LOCAL_FILENAME,
            "text/plain",
            "Summarize the plot of Hamlet in two sentences.",
        ],
        id=f"ai-claimcheck-rag-{int(time.time())}",
        task_queue="claim-check-pattern-task-queue",
    )

    print("=== AI RAG with Claim Check ===")
    print("Answer:", result.answer)
    print("Sources:", result.sources)


if __name__ == "__main__":
    asyncio.run(main())
