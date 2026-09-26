# DocuChat

DocuChat is a local-first web application for chatting with PDF documents using retrieval-augmented generation (RAG). It uploads text-based PDFs, chunks and embeds their content with local Ollama models, searches using vector and lexical signals, and answers with page citations.

## Stack and direction

- Frontend: Next.js, React, TypeScript, hand-written CSS
- Backend: FastAPI and Python
- Local AI: Ollama with `qwen3:8b` for generation and `qwen3-embedding:0.6b` for embeddings
- Storage: SQLite for zero-config local development, PostgreSQL supported through `DATABASE_URL`, and ChromaDB for vectors

No OpenAI API, cloud LLM service, or Streamlit is used.

## Repository layout

```text
docuchat/
├── frontend/       Next.js UI
├── backend/        FastAPI application
├── data/           Local runtime data (ignored by Git)
├── evaluation/     Future RAG evaluation assets
└── scripts/        Future repeatable development scripts
```

The backend keeps HTTP routes in `app/api`, configuration in `app/core`, and the working RAG pipeline in `app/services`: `documents.py` (PDF extraction, chunking, embedding), `retrieval.py` (hybrid vector + lexical search, plus the whole-document "overview" retrieval used for summary-style questions), `ollama.py` (the only place that talks to the local model runtime), and `vector_store.py` (the shared ChromaDB collection). A future image/chart/scan pipeline can be added as a separate service module, so it does not become coupled to text extraction or the generation model.

## Prerequisites

- Node.js 20 or newer
- Python 3.11 or newer
- Ollama, with `qwen3:8b` and `qwen3-embedding:0.6b` pulled

## First-time setup

From the repository root:

```bash
cp .env.example .env
python3 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
cd frontend
npm install
```

Install and start Ollama, then download the two local models:

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

The default configuration uses SQLite, so the app works without a database server. To use PostgreSQL instead, start the included service and set `DATABASE_URL` in `.env`:

```bash
docker compose up -d postgres
DATABASE_URL=postgresql+psycopg://docuchat:docuchat_dev_password@127.0.0.1:5432/docuchat
```

## Run the backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

`python -m app.main` (from `backend/`, with the venv active) also works as a one-command alternative — it reads `BACKEND_HOST`/`BACKEND_PORT` from `.env` and enables auto-reload.

Open `http://127.0.0.1:8000/docs` for the automatic API documentation. The health endpoint is `GET /api/v1/health`.

## Run the frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## Verify everything works

```bash
cd backend
source .venv/bin/activate
pytest
curl http://127.0.0.1:8000/api/v1/health
```

Expected response:

```json
{"status":"ok","service":"docuchat-api"}
```

## Current capabilities

- PDF upload, text extraction, and local persisted document records; ingestion (extraction → chunking → embedding) runs as a background task, so the upload request returns immediately and the UI polls for status
- Duplicate uploads are detected by content hash and rejected with a clear message instead of re-ingesting the same PDF twice
- An upload size limit (`MAX_UPLOAD_SIZE_MB`, default 50) rejects oversized files before they're processed
- Page-aware chunking and local Qwen embeddings in ChromaDB, embedded in parallel batches
- Hybrid retrieval for specific questions: vector search supplemented by lexical matching, fused with a reciprocal-rank-fusion-style formula
- A separate whole-document "overview" retrieval path for summary-style questions ("summarize this", "what is this document about", "list the topics covered") that a narrow similarity search can't answer, since no single passage *is* a summary
- Local Qwen generation with source markers and expandable page citations; the model's internal `<think>` reasoning trace is skipped by default for much faster answers (`OLLAMA_GENERATION_THINK`)
- Short conversation history is included as context (not evidence) so follow-up questions can resolve references like "what about the second point"
- Document selection, removal, and persistent conversation records
- A PDF with no extractable text (e.g. a scanned image) is marked `empty` rather than a misleading `ready`
- Clear UI status for unavailable Ollama models

## Intentionally deferred

Scanned PDFs, OCR, table/chart extraction, vision models, authentication, multi-user permissions, streaming responses, a trained reranker model, and production deployment hardening remain future work. The services layer deliberately leaves space for an independent local vision/OCR processor; no vision model has been selected or downloaded.
