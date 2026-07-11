# rag-service

RAG service built with FastAPI, Pydantic v2, Anthropic Claude (generation), and
OpenAI `text-embedding-3-small` (retrieval). Retrieval is hand-rolled — no
LangChain/LlamaIndex — so the chunk/embed/search mechanics stay visible.

## Setup

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
uv sync
```

Drop `.txt`/`.md` files into `data/` — they're ingested automatically on
server startup.

## Run

```bash
uv run uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000` and ingests everything in
`data/` into an in-memory vector store before accepting requests.

## Ingest standalone

To run the chunk/embed pipeline without starting the server (useful for
checking chunk counts or embedding call behavior):

```bash
uv run python scripts/ingest.py --data-dir data
```

This uses its own throwaway `InMemoryStore` — it doesn't feed the running
server's store, since that store is rebuilt in-process at startup.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/query` | Ask a question |

### POST /query

```json
// request
{ "question": "What is RAG?", "top_k": 4 }

// response
{
  "answer": "...",
  "sources": [{ "doc_id": "rag-service", "chunk_id": 0, "score": 0.83 }],
  "grounded": true
}
```

`top_k` (default `4`) controls how many chunks are retrieved and handed to
the LLM as context — higher means better recall but noisier/more expensive
prompts. `grounded` is `true` whenever at least one chunk was retrieved; it
does not (yet) check that the chunk is actually relevant.

## What happens on a `/query` call

1. **Embed the question** — `EmbeddingClient.embed()` sends it to OpenAI and
   gets back a vector.
2. **Retrieve** — `InMemoryStore.search()` compares that vector against every
   stored chunk vector via cosine similarity and returns the top `k`.
3. **Augment** — the retrieved chunks are concatenated into a `context`
   string and inserted into the prompt, tagged with their `doc_id`/`chunk_id`
   so the model (and the response's `sources` field) can cite where they
   came from.
4. **Generate** — `LLMClient.generate()` sends the system prompt + context +
   question to Claude and returns the answer text.

## Project layout

```
app/
  config.py       # pydantic-settings — all runtime config, read from .env
  schemas.py      # request / response models (QueryRequest, QueryResponse)
  embeddings.py   # EmbeddingClient — wraps OpenAI embeddings API
  store.py        # VectorStore Protocol + InMemoryStore (numpy cosine similarity)
  ingest.py       # load data/, chunk, embed, upsert — the ingestion pipeline
  llm.py          # LLMClient + AnthropicBackend — wraps Claude generation
  main.py         # FastAPI app; ingests data/ on startup, wires /query to retrieval
scripts/
  ingest.py       # standalone CLI wrapper around app/ingest.py
data/             # source documents (.txt / .md), ingested on startup
```

### `app/config.py`

A single `Settings` object (pydantic-settings) read once from `.env`. Every
tunable in the system lives here: API keys, model names, chunking
parameters (`chunk_tokens`, `chunk_overlap`), `data_dir`, and `log_level`.
Nothing else in the codebase reads environment variables directly.

### `app/embeddings.py`

`EmbeddingClient` wraps OpenAI's embeddings endpoint. Two methods:
`embed(text)` for a single string (used for the incoming question) and
`embed_batch(texts)` for many at once (used during ingestion, since batching
is cheaper and faster than one call per chunk). Logs a summary (model,
input count, token usage) at `INFO`, and the full raw API response at
`DEBUG`.

### `app/store.py`

Defines the storage contract as a `VectorStore` Protocol —
`upsert(chunks)` / `search(query_vec, top_k)` — so the backing store is
swappable without touching ingestion or query code. `InMemoryStore` is the
only implementation: it keeps chunks in a dict and their embeddings in a
normalized numpy matrix, and `search()` is a single matrix-vector dot
product (cosine similarity, since everything is pre-normalized) followed by
a top-k partition. Fine up to tens of thousands of chunks; swap in
pgvector/Qdrant/etc. past that.

### `app/ingest.py`

The ingestion pipeline, callable both from `main.py` (on server startup) and
`scripts/ingest.py` (standalone):

- `load_documents(data_dir)` reads every `.txt`/`.md` file, keyed by
  filename stem as `doc_id`.
- `chunk_text(text, doc_id)` splits a document into overlapping,
  token-bounded chunks (~600 tokens, ~80 overlap by default) using the same
  tokenizer OpenAI's embedding models use, so chunk size is measured the
  way the model actually sees it rather than by character/word count.
- `ingest_directory(...)` ties it together: load → chunk → embed (via
  `EmbeddingClient.embed_batch`) → `store.upsert(...)`.

### `app/llm.py`

Mirrors `store.py`'s swappable-backend pattern for generation:
`LLMBackend` is an abstract base, `AnthropicBackend` is the only
implementation (streams a response from Claude and returns the final text),
and `create_llm_client` picks a backend by `settings.llm_provider`. Add a
new provider by subclassing `LLMBackend` and registering it in the
`backends` dict.

### `app/main.py`

The FastAPI app. On startup (`lifespan`), it builds the `LLMClient`,
`EmbeddingClient`, and a fresh `InMemoryStore`, then runs
`ingest_directory` against `data_dir` so the store is populated before the
server accepts traffic. `/query` runs the four-step flow described above.
Logging is configured here too (`logging.basicConfig`, level from
`settings.log_level`) — uvicorn's own logging setup doesn't configure the
root logger, so without this, `logger.info`/`.debug` calls elsewhere in the
app would be silently dropped.

### `scripts/ingest.py`

A thin CLI (`argparse`) around `ingest_directory`, for running the
chunk/embed pipeline in isolation — e.g. to check chunk counts or watch
embedding calls without starting the server. It builds its own throwaway
`InMemoryStore`, so it does **not** feed the running server; the server
always ingests its own copy at startup.

## Adding a new LLM provider

1. Subclass `LLMBackend` in `app/llm.py` and implement `generate(system, user)`.
2. Register the new class in the `backends` dict inside `create_llm_client`.
3. Set `LLM_PROVIDER=<your-key>` in `.env`.

## Adding a new vector store backend

1. Implement the `VectorStore` Protocol (`upsert`, `search`) in a new class
   in `app/store.py` (or a new module).
2. Swap the `InMemoryStore()` instantiation in `app/main.py`'s `lifespan`
   for your implementation.
