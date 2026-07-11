# RAG Service Overview

This service answers questions by retrieving relevant chunks from documents
placed in the `data/` directory and passing them to an LLM as context.

## Retrieval pipeline

1. Documents in `data/` are split into ~600-token chunks with ~80-token
   overlap, so no fact is severed cleanly at a chunk boundary.
2. Each chunk is embedded with OpenAI's `text-embedding-3-small` model.
3. Embeddings are stored in an in-memory vector store that ranks chunks by
   cosine similarity.
4. At query time, the question is embedded and the top-k most similar
   chunks are retrieved and included as context for the LLM.

## Swapping components

The vector store is defined as a `VectorStore` Protocol, so the in-memory
implementation can be swapped for pgvector, Qdrant, or another backend
without touching the ingestion or query code.
