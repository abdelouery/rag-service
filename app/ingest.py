from pathlib import Path

import tiktoken

from .embeddings import EmbeddingClient
from .store import Chunk, VectorStore

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_documents(data_dir: Path) -> dict[str, str]:
    """Reads .txt/.md files from data_dir, keyed by doc_id (filename stem)."""
    docs: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES:
            docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def chunk_text(
    text: str,
    doc_id: str,
    chunk_tokens: int = 600,
    overlap: int = 80,
) -> list[Chunk]:
    """Splits text into overlapping, token-bounded chunks.

    Tokenizing with the same encoder OpenAI uses keeps chunks under the
    embedding model's context window regardless of the source language or
    markup density (code/CJK text can pack far more characters per token
    than plain English).
    """
    # Work in token space, not characters/words, so chunk size matches what
    # the embedding model actually sees.
    tokens = _ENCODING.encode(text)

    # Each new chunk starts `stride` tokens after the previous one, not
    # `chunk_tokens` after — that's what creates the overlap. E.g. with
    # chunk_tokens=600, overlap=80: chunk 0 covers [0:600], chunk 1 starts
    # at 520 and covers [520:1120], etc.
    stride = chunk_tokens - overlap

    chunks: list[Chunk] = []
    chunk_id = 0
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        piece = _ENCODING.decode(tokens[start:end])  # tokens back to text
        chunks.append(
            Chunk(
                id=f"{doc_id}::{chunk_id}",
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=piece,
                metadata={"doc_id": doc_id, "chunk_id": chunk_id},
            )
        )
        chunk_id += 1
        if end == len(tokens):
            break  # just emitted the last (possibly short) chunk
        start += stride

    return chunks


async def ingest_directory(
    data_dir: Path,
    embedder: EmbeddingClient,
    store: VectorStore,
    chunk_tokens: int = 600,
    overlap: int = 80,
) -> int:
    """Loads, chunks, embeds, and upserts every document in data_dir. Returns chunk count."""
    # 1. Load every source file into memory.
    documents = load_documents(data_dir)

    # 2. Split each document into overlapping chunks (still no embeddings yet).
    all_chunks: list[Chunk] = []
    for doc_id, text in documents.items():
        all_chunks.extend(chunk_text(text, doc_id, chunk_tokens, overlap))

    if not all_chunks:
        return 0

    # 3. Embed every chunk's text in one batched API call (cheaper/faster
    # than one call per chunk), then attach each vector back to its chunk.
    embeddings = await embedder.embed_batch([c.text for c in all_chunks])
    for chunk, vector in zip(all_chunks, embeddings):
        chunk.embedding = vector

    # 4. Hand the fully-embedded chunks to the store for indexing.
    store.upsert(all_chunks)
    return len(all_chunks)
