# DocuChat

DocuChat is a local-first web application for chatting with PDF documents using retrieval-augmented generation (RAG). It uploads PDFs (including scanned ones, via local OCR), chunks and embeds their content with local Ollama models, searches using vector and lexical signals, and answers with page citations.

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

The backend keeps HTTP routes in `app/api`, configuration in `app/core`, and the working RAG pipeline in `app/services`: `documents.py` (format-independent ingestion: chunking, embedding, indexing, document status), `retrieval.py` (hybrid vector + lexical search, plus the whole-document "overview" retrieval used for summary-style questions), and `vector_store.py` (the shared ChromaDB collection).

Three seams keep the rest of the code independent of any one file format or model runtime:

- **Documents** (`document_model.py`, `pdf_provider.py`): a `Document → Pages → Blocks → Chunks` model and a `DocumentProvider` interface. `PdfDocumentProvider` is the only implementation and the only module that touches PyMuPDF; it handles native text extraction, the OCR fallback, figure metadata and page rendering.
- **Models** (`providers/`): `LLMProvider`, `EmbeddingProvider` and (optional) `VisionProvider` interfaces with Ollama implementations. `services/ollama.py` is a thin facade (`generate`, `embed`) over whichever provider is installed, so services need no changes and tests can install fakes with `use_providers(...)`.
- **OCR** (`ocr.py`): an `OcrEngine` interface with a local RapidOCR implementation.

## Scanned PDFs, figures and prompt templates

- **OCR.** A page is sent to OCR only when its native text layer is thin *and* images cover most of it; pages with real text are never OCR'd, and a document is capped at `OCR_MAX_PAGES` OCR'd pages. Each page gets an explicit status (`text`, `ocr`, `empty`, `failed`) and OCR'd text is indexed by the same pipeline as native text (vectors, FTS5, citations, Tutor/Exam/Brainstorm), always cited to its real PDF page. If OCR isn't installed or fails, affected pages are reported as failed with setup guidance, never silently treated as empty. Install with `pip install -r backend/requirements.txt`.
- **Figures.** Embedded images and "Figure N:" captions are recorded as metadata (never pixel data in the database); page and figure previews are rendered on demand into `data/processed`. Questions about a chart or diagram are answered honestly: with no local vision model configured (`OLLAMA_VISION_MODEL`), the reply says visual Q&A isn't supported and points to the page, figure preview and extracted text instead of pretending to read the image.
- **Prompt templates.** Six built-in templates (Research Paper Review, Case Brief, Lecture Notes, Executive Summary, Study Guide, Technical Documentation Review) plus simple custom ones (name, description, instruction, optional output format). Applying one runs through the normal retrieval, grounding and citation pipeline over the selected document or collection.

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
- Clear UI status for unavailable Ollama models, naming which model is missing

## Document states

A document is always in exactly one state, shown in the library and driven by the same `status` / `status_detail` the API returns:

| State | Meaning | What you can do |
| --- | --- | --- |
| Processing | Being read (live stage: "Extracting text", "OCR processing page 2 (1 of 4)", "Indexing") | Open it in the viewer; chat once it's Ready |
| Ready | Indexed and searchable (a "N OCR" badge marks scanned pages read by OCR) | Chat, search, contents, figures |
| No readable text (`empty`) | Every page is blank, or an image OCR couldn't read | Remove it |
| Failed | Password-protected, damaged, models unavailable, or interrupted by an app restart -- the reason is shown | Remove it and upload again |

Refused at upload, so it never enters the library: not a PDF (415), empty (400), over the size limit (413), or already in the library (409, naming the existing document). A document still Processing when the app stops is marked Failed on the next start rather than spinning forever.

## Answering when the document doesn't say

Every mode is meant to fail closed. Beyond the model's own instructions, plain code checks what it can: a page that doesn't exist (or has no text) is reported from the page count; a "where is X discussed" answer only names pages that cover the whole topic; calculations never use a year as an operand; extraction values must appear in the passage they cite (otherwise "Not stated", and an entity found in no passage is dropped); a quiz topic the document never mentions is refused; and a template written for one kind of document (Case Brief) is refused for another. Go Freely still allows general knowledge, but only under its own uncited "Additional context" heading.

## Local model notes

- `OLLAMA_NUM_CTX` (default 8192) is sent with every generation. Ollama's default 4096-token window silently truncates longer prompts from the front, which drops the instructions -- see `.env.example`.
- On a 16 GB Mac the text model and the vision model don't both stay loaded: a chart question evicts `qwen3:8b`, and the next text question pays a ~7 s reload (measured). Ordinary text questions never touch the vision model.
- `OLLAMA_EMBED_CONCURRENCY` is bounded parallelism, but on that hardware embedding is compute-bound and extra concurrency gave no speedup.

## Intentionally deferred

Table/chart *extraction*, authentication, multi-user permissions, streaming responses, a trained reranker model, and production deployment hardening remain future work. Vision is limited to answering a question about one rendered page/figure with a locally installed vision model (`OLLAMA_VISION_MODEL`); nothing is downloaded for you.
