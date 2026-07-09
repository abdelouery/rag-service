# rag-service

RAG service skeleton built with FastAPI, Pydantic v2, and Anthropic Claude. No retrieval yet — this is the scaffold.

## Setup

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`.

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
{ "answer": "...", "sources": [], "grounded": true }
```

## Project layout

```
app/
  config.py    # pydantic-settings (reads .env)
  schemas.py   # request / response models
  llm.py       # LLMClient + AnthropicBackend (swappable)
  main.py      # FastAPI app, routes
```

## Adding a new LLM provider

1. Subclass `LLMBackend` in `app/llm.py` and implement `generate(system, user)`.
2. Register the new class in the `backends` dict inside `create_llm_client`.
3. Set `LLM_PROVIDER=<your-key>` in `.env`.
