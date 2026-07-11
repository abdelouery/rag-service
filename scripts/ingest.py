"""Standalone CLI to run the ingestion pipeline against data/.

Note: app.main builds its own InMemoryStore in-process at server startup
(so /query has something to search), so this script doesn't feed the
running server — it's for validating the pipeline (chunk counts, embedding
calls, timing) independently of the API.
"""

import argparse
import asyncio
from pathlib import Path

from app.config import Settings
from app.embeddings import EmbeddingClient
from app.ingest import ingest_directory
from app.store import InMemoryStore


async def run(data_dir: Path, chunk_tokens: int, overlap: int) -> None:
    settings = Settings()
    embedder = EmbeddingClient(settings)
    store = InMemoryStore()

    count = await ingest_directory(data_dir, embedder, store, chunk_tokens, overlap)
    print(f"Ingested {count} chunks from {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--chunk-tokens", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    asyncio.run(run(args.data_dir, args.chunk_tokens, args.overlap))


if __name__ == "__main__":
    main()
