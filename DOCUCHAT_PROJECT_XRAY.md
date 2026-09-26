# DocuChat Project X-Ray

A complete, beginner-friendly guide to what you built, how it actually works, and how to make it better. Every claim in this document is based on reading the real code in this repository on 2026-09-18, not on assumptions from file names or the README. Where the README says one thing and the code does another, that's called out explicitly.

---

## PART 1 — The Big Picture

### 1. What is DocuChat?

DocuChat is a website where you upload a PDF and then ask questions about it in a chat window, and the answers are grounded in that PDF's actual text (with page citations), not made up from the AI's general knowledge.

It runs entirely on your own machine. There's no OpenAI, no Anthropic API, no cloud database. The two AI models (one for "understanding" text, one for "answering") run locally through a tool called **Ollama**.

### 2. What problem does it solve?

Normally, if you paste a 30-page PDF into ChatGPT and ask a question, one of a few things happens:
- The document is too long to fit in the model's context window.
- The model "skims" and gives a vague or wrong answer.
- You have no way to check *which part* of the document the answer came from.
- Your document leaves your machine and goes to a third-party server.

DocuChat solves this with a technique called **RAG (Retrieval-Augmented Generation)**: instead of feeding the whole PDF to the AI, it finds the *specific paragraphs* that are relevant to your question and only shows the AI those. This keeps answers grounded, keeps costs/compute low, and lets you cite exactly where an answer came from — all without sending anything to the cloud.

### 3. What can a user currently do with it?

Based on the actual working code (not aspirational features):

- Upload a text-based PDF (a PDF where you can select/copy the text — not a scanned image).
- See it appear in a "library" sidebar once it's processed.
- Select one or more uploaded documents (or none, more on that below).
- Ask a question in a chat box.
- Get back an answer with clickable citation markers like `[1]`, `[2]` that expand to show the exact page and text excerpt used.
- Start a "new conversation" and have prior conversations persist in the database.
- Delete a document (which also removes its data from the vector database).
- See a status indicator showing whether the local AI (Ollama) is running and has the right models downloaded.

What it can **not** do (confirmed by absence of any related code): read scanned/image-only PDFs (no OCR), understand images/charts inside PDFs, stream answers token-by-token, log in as different users, or run without Ollama installed locally.

### 4. What happens from the moment the app starts until the user gets an answer?

At a high level:

1. You start two separate processes: the **backend** (FastAPI, Python, port 8000) and the **frontend** (Next.js, React, port 3000).
2. You open `http://localhost:3000` in a browser. The React page loads and immediately asks the backend "what documents exist?" and "is Ollama ready?"
3. You upload a PDF. The browser sends the raw file to the backend over HTTP. The backend saves it, extracts text with a PDF-reading library, splits the text into small chunks, converts each chunk into a list of numbers (an **embedding**) using a local AI model, and stores those chunks + numbers in a vector database (**ChromaDB**).
4. You type a question. The browser sends it to the backend. The backend turns your question into an embedding too, searches the vector database for the most similar chunks, also runs a plain keyword search as a backup signal, blends the two rankings together, and picks the best ~8 chunks.
5. The backend builds a prompt that says "here are some passages, answer ONLY using them" and sends that to a local chat model (also via Ollama). The model replies with a JSON answer plus which passages it used.
6. The backend cleans up that response, builds citation objects (filename + page + excerpt), saves the conversation to the database, and sends the answer back to the browser.
7. React renders the answer as chat bubbles with expandable citations underneath.

### 5. What technologies are used?

| Layer | Technology | Plain-English role |
|---|---|---|
| Frontend | Next.js 15 + React 19 + TypeScript | The webpage you interact with |
| Styling | Hand-written CSS (Tailwind is installed but barely used — see Part 18) | Makes the page look nice |
| Backend | FastAPI (Python) | The web server that handles uploads and questions |
| Relational database | SQLite (default) / PostgreSQL (optional) via SQLAlchemy | Stores documents, chat history, and chunk text |
| Vector database | ChromaDB | Stores the "meaning fingerprints" (embeddings) of each text chunk so similar ones can be found fast |
| PDF reading | PyMuPDF (`fitz`) | Extracts raw text from PDF pages |
| Local AI runtime | Ollama | Runs the two AI models on your own computer, exposed as a local HTTP API |
| Generation model | `qwen3:8b` (via Ollama) | Reads the retrieved passages and writes the answer |
| Embedding model | `qwen3-embedding:0.6b` (via Ollama) | Converts text into numbers for similarity search |

**Plain-English definitions used above:**
- **Embedding** — converting text into a list of numbers (a vector) so a computer can measure how similar two pieces of text are in *meaning*, not just spelling.
- **Vector database** — a database built specifically to store these lists of numbers and quickly answer "which stored items are most similar to this new item?"
- **Local AI runtime (Ollama)** — a background program that downloads and runs open-weight AI models on your own hardware, and lets other programs (like this backend) talk to those models over `http://127.0.0.1:11434`, the same way you'd talk to OpenAI's API — except nothing leaves your machine.

---

## PART 2 — DocuChat: The Story of One Question

Let's follow one real document that already exists in this project's data folder: `Presentation_Prep_Complete_Walkthrough.pdf`, a 9-"page" document that, when inspected, actually turned out to contain 1,407 text chunks — meaning it's an extremely text-dense document (each PDF "page" here holds far more text than a typical printed page). This is real data sitting in the project's SQLite database and ChromaDB store right now, not a hypothetical.

**Uploading it:**

> A user drags this PDF onto the upload button. The browser immediately sends it as a raw file (`multipart/form-data`) to `POST /api/v1/documents`. The backend does not ask the user anything else — no title, no tags, no folder. It generates a random ID for the document, saves the raw PDF to disk under that ID (e.g. `542475a0-...pdf`), and creates a database row with `status="processing"`.
>
> Then, still inside that same HTTP request (nothing happens in the background), the backend opens the PDF with PyMuPDF, walks through each of its 9 pages, pulls out the plain text, and slices each page's text into ~900-character pieces with 150 characters of overlap between consecutive pieces (so a sentence that gets cut in half at a chunk boundary still appears in full in the *next* chunk too). For this document, that produced roughly 150+ chunks per page.
>
> Each chunk is sent to the local embedding model (`qwen3-embedding:0.6b`, running inside Ollama) in batches of 16, which converts it into a 1024-ish-dimension list of numbers. Those numbers, plus the raw text and metadata (which document, which page), are stored in ChromaDB. The chunk text is *also* stored in SQLite, because the vector database is only good at "find similar," not at "fetch by document ID for a normal listing" — the two databases play different roles.
>
> Finally the document's status flips to `"ready"` and the HTTP response returns to the browser, which refreshes the sidebar list. The whole thing — extraction, chunking, embedding 1,400+ chunks — happens synchronously, so for a document this dense, the upload button would stay in its "Reading your PDF…" state for a while.

**Asking a question — "What is overfitting?":**

> The user selects the document's checkbox (or doesn't — selection isn't actually enforced, see Part 18) and types "What is overfitting?" into the composer, then hits send.
>
> The browser calls `POST /api/v1/chat` with `{question, document_ids, conversation_id}`. If there's no `conversation_id` yet, the backend creates a new `Conversation` row titled with the first 70 characters of the question.
>
> The backend does **not** send the raw question straight to the chat model. First it goes to `retrieve()`:
> 1. It loads all chunk rows for the selected document(s) — or *every* document in the database if none were explicitly selected — from SQLite.
> 2. It builds a special embedding-search string: `"Instruct: Given a document question, retrieve the passage that directly answers it.\nQuery: What is overfitting?"` — this exact instruction-prefix format is what the Qwen3 embedding model expects for asymmetric search (question vs. passage), not a rewrite of your question by an LLM.
> 3. It embeds that string and asks ChromaDB for the 24 nearest chunks by cosine similarity.
> 4. Separately, it runs a hand-written keyword scorer over the *same* candidate chunks — counting overlapping words and rewarding exact 2-3 word phrase matches.
> 5. It merges the two ranked lists using a formula (`1 / (60 + rank)`), a lightweight version of a well-known trick called **Reciprocal Rank Fusion** — chunks that rank highly in *either* the vector search or the keyword search bubble to the top.
> 6. It keeps the top 8 chunks.
>
> If somehow zero chunks come back (e.g., a scanned PDF with no extractable text, or no documents uploaded at all), the backend immediately returns an HTTP 400 error: *"Upload a text-based PDF before asking a question."* No AI model is even called in that case.
>
> Otherwise, the 8 chunks get formatted into a block like:
> ```
> SOURCE 1 | Presentation_Prep_Complete_Walkthrough.pdf | page 3
> Overfitting happens when a model learns noise in the training data...
> ```
> This block is glued into a strict instruction prompt: *"Answer using ONLY the supplied passages... Return exactly one JSON object... Use 1 to 3 source_ids."* That whole prompt, plus a system message ("You are a strict evidence-grounded research assistant. Never use outside knowledge."), is sent to `qwen3:8b` through Ollama's `/api/chat` endpoint.
>
> The model — which is a "thinking" model that emits internal reasoning wrapped in `<think>...</think>` tags before its real answer — replies. The backend strips out the `<think>` block, extracts the JSON object, checks that every `source_id` the model claims actually exists among the 8 sources, discards anything outside that range, keeps at most 3, and force-appends a `[n]` marker into the answer text if the model forgot one. If the model's JSON is malformed or missing citations entirely, the backend falls back to plain-text parsing with a regex, and if it *still* can't find a valid, cited answer, it replaces whatever the model said with a fixed, honest fallback: *"I couldn't verify a source-grounded answer from the retrieved document passages."*
>
> Finally, both the user's question and the assistant's answer are saved as `Message` rows (with the citations serialized as JSON text), and the response — answer text plus a `citations` array (index, filename, page number, and a 280-character excerpt) — goes back to the browser, which renders it as a chat bubble with expandable `<details>` citation blocks underneath.

This *is* what the code actually does — it is a real hybrid-retrieval RAG pipeline, not just a "stuff the whole PDF into the prompt" system.

---

## PART 3 — Complete Project Structure

```text
docuchat/  (repo root: "referenced-chatgpt-conversation-this-is-an/")
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entrypoint, wires routers + CORS + DB init
│   │   ├── db.py                    # SQLAlchemy engine/session setup
│   │   ├── models.py                # ORM tables: Document, Chunk, Conversation, Message
│   │   ├── core/
│   │   │   └── config.py            # Settings loaded from .env (pydantic-settings)
│   │   ├── api/v1/
│   │   │   ├── health.py            # GET /api/v1/health
│   │   │   ├── system.py            # GET /api/v1/system/status (Ollama readiness)
│   │   │   ├── documents.py         # Upload / list / delete PDFs
│   │   │   └── chat.py              # Ask questions, list/get conversations
│   │   └── services/
│   │       ├── documents.py         # PDF extraction, chunking, embedding, ingestion pipeline
│   │       ├── ollama.py            # HTTP client for the local Ollama API (embed/generate/status)
│   │       ├── retrieval.py         # Hybrid vector + lexical search and fusion ranking
│   │       └── vector_store.py      # ChromaDB client/collection setup
│   ├── tests/
│   │   └── test_health.py           # One test: health endpoint
│   ├── requirements.txt             # Runtime Python dependencies
│   └── requirements-dev.txt         # + pytest for testing
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 # The entire UI: sidebar + chat, in one component
│   │   ├── layout.tsx               # Next.js root HTML shell + page metadata
│   │   └── globals.css              # All hand-written styling (+ unused Tailwind setup)
│   ├── lib/
│   │   └── api.ts                   # Typed fetch wrapper — the only place that calls the backend
│   ├── public/
│   │   └── docuchat-ai-logo.png     # Unused image asset (see Part 18)
│   ├── package.json / tsconfig.json / next.config.ts / tailwind.config.ts / postcss.config.mjs
├── data/                             # Runtime data, git-ignored except folder markers
│   ├── docuchat.db                  # SQLite database file
│   ├── chroma/                      # ChromaDB's persisted vector index files
│   ├── uploads/                     # Raw uploaded PDFs, named "{document_id}.pdf"
│   └── processed/                   # Empty placeholder — nothing in the code writes here
├── evaluation/
│   └── README.md                    # Placeholder only — no evaluation code exists yet
├── scripts/
│   └── README.md                    # Placeholder only — no scripts exist yet
├── docker-compose.yml                # Optional local PostgreSQL container (NOT the whole app)
├── .env / .env.example
├── .gitignore
└── README.md
```

**Note on `docker-compose.yml`** (the file you had open): it does **not** containerize DocuChat. It only spins up a bare PostgreSQL database as an *alternative* to SQLite. There is no `Dockerfile` anywhere in the repo, so the backend and frontend themselves are never run inside Docker — you always run them directly with `uvicorn` and `npm run dev`.

Also worth knowing before we go further: **this directory is not a git repository** (there's no `.git` folder). None of this work has version history yet — worth doing before treating this as a portfolio piece (see Part 19).

---

## PART 4 — File-by-File Explanation

| File | Purpose | Used By | Calls | Key Functions/Classes | Essential? |
|---|---|---|---|---|---|
| `backend/app/main.py` | Creates the FastAPI app, registers CORS and all routers, creates DB tables on startup | Run via `uvicorn app.main:app` | `app.api.v1.*`, `app.db`, `app.core.config` | `create_database_tables()` | Essential |
| `backend/app/core/config.py` | Reads `.env` into a typed settings object | Every backend module that needs config | — | `Settings`, `get_settings()` | Essential |
| `backend/app/db.py` | Sets up the SQLAlchemy engine, session factory, declarative base | `models.py`, all API routes via `get_db` | `core/config.py` | `get_db()` | Essential |
| `backend/app/models.py` | Defines the 4 database tables | `db.py` (table creation), all API routes, services | — | `Document`, `Chunk`, `Conversation`, `Message` | Essential |
| `backend/app/api/v1/health.py` | Liveness check, no dependency checks | Frontend? No — actually unused by the frontend (see Part 18) | nothing | `health_check()` | Optional (dev/ops convenience) |
| `backend/app/api/v1/system.py` | Reports whether Ollama is reachable and has the right models pulled | Frontend sidebar status card | `services/ollama.status()` | `system_status()` | Essential (drives real UI state) |
| `backend/app/api/v1/documents.py` | Upload, list, delete PDFs | Frontend `lib/api.ts` | `services/documents.process_document`, `services/vector_store.get_collection` | `list_documents`, `upload_document`, `delete_document` | Essential |
| `backend/app/api/v1/chat.py` | Handles asking questions and conversation history | Frontend `lib/api.ts` | `services/retrieval.retrieve`, `services/ollama.generate` | `ask`, `grounded_response`, `list_conversations`, `get_conversation` | Essential |
| `backend/app/services/documents.py` | The ingestion pipeline: extract → chunk → embed → store | `api/v1/documents.py` | `fitz` (PyMuPDF), `services/ollama.embed` | `chunk_text`, `embed_in_batches`, `process_document` | Essential |
| `backend/app/services/ollama.py` | The only place that talks HTTP to Ollama | `services/documents.py`, `services/retrieval.py`, `api/v1/chat.py`, `api/v1/system.py` | Ollama's local HTTP API | `embed`, `generate`, `status`, `OllamaUnavailable` | Essential |
| `backend/app/services/retrieval.py` | Hybrid vector+keyword search and score fusion | `api/v1/chat.py` | `services/ollama.embed`, ChromaDB collection, SQLite via SQLAlchemy | `retrieve`, `lexical_score`, `terms` | Essential |
| `backend/app/services/vector_store.py` | Opens/creates the ChromaDB collection | `api/v1/documents.py`, `api/v1/chat.py` (both call `get_collection()` fresh each time) | `chromadb` | `get_collection()` | Essential |
| `backend/tests/test_health.py` | Confirms the app boots and `/health` responds | `pytest` | `app.main` | `test_health_check` | Optional but cheap to keep |
| `frontend/app/page.tsx` | The entire UI: upload zone, document library, chat thread, composer | Loaded by Next.js as the home route | `lib/api.ts` functions | `Home()`, `FormattedText()` | Essential |
| `frontend/lib/api.ts` | Typed fetch wrapper; the single seam between UI and backend | `page.tsx` | Backend HTTP endpoints | `request<T>`, `listDocuments`, `uploadDocument`, `deleteDocument`, `askQuestion`, `getSystemStatus`, `getHealth` (unused) | Essential (minus the one dead export) |
| `frontend/app/layout.tsx` | Root HTML wrapper + page `<title>`/description | Next.js framework | `globals.css` | `RootLayout()` | Essential (framework requirement) |
| `frontend/app/globals.css` | All visual styling, hand-written and minified | `layout.tsx` import | Tailwind base layers (barely) | — | Essential, but see Part 18 for unused rules |
| `frontend/public/docuchat-ai-logo.png` | A logo image | Nothing currently references it | — | — | **Appears unused** |
| `data/processed/` | Empty folder | Nothing reads/writes it | — | — | **Appears unused / dead placeholder** |
| `evaluation/`, `scripts/` | Placeholder folders for future work | Nothing yet | — | — | Not yet functional — intentionally empty per README |
| `docker-compose.yml` | Optional Postgres dev database | Manually via `docker compose up -d postgres` | — | — | Optional |

### "Think of this file as…"

- **`backend/app/main.py`** — the building's front desk. It doesn't do any real work itself; it just makes sure every department (documents, chat, health, system) has a listed extension and knows how to answer the door (CORS).
- **`backend/app/services/documents.py`** — the intake clerk who receives a new book, tears it into index cards (chunks), and files each card with a fingerprint (embedding) so it can be found later.
- **`backend/app/services/retrieval.py`** — the librarian who, when you ask a question, doesn't reread every book — instead checks the card catalog twice (once by "meaning," once by keyword) and hands you the eight most promising cards.
- **`backend/app/services/ollama.py`** — the translator/courier between DocuChat and the two AI models running on your machine. Every request to an AI model passes through this one file.
- **`backend/app/api/v1/chat.py`** — the strict editor. It takes the librarian's cards and the AI's draft answer and refuses to publish anything that isn't backed by a citation.
- **`frontend/lib/api.ts`** — the receptionist between the browser and the backend: every button click that needs backend data goes through this one file, never directly.
- **`frontend/app/page.tsx`** — the entire storefront: reception desk, bookshelf, and reading room, all built as one big room (one React component) rather than separate rooms (separate components).

---

## PART 5 — Application Startup Flow

There is exactly **one intended way** to run this project locally (per `README.md`), and it requires two terminals plus Ollama running in the background. There is no single "start everything" script and no Docker path for the app itself.

1. **Command that starts the backend:** `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`, run from inside `backend/` with its virtualenv activated.
2. **First file that runs:** `app/main.py` is imported by Uvicorn.
3. **What gets initialized, in order, as `main.py` executes top-to-bottom:**
   - `get_settings()` reads `../.env` (relative to `backend/`, i.e. the repo-root `.env`) into a `Settings` object — this is cached (`@lru_cache`) so it's only parsed once per process.
   - A `FastAPI` app instance is created with title/version from settings.
   - CORS middleware is added, restricted to `settings.cors_origin_list` (defaults to `http://localhost:3000`).
   - Four routers are mounted under the `/api/v1` prefix: health, documents, chat, system.
   - The SQLAlchemy `engine` (from `db.py`) is created eagerly at import time — this is the moment the SQLite file path (or Postgres connection string) is resolved.
4. **On the `startup` event** (`@app.on_event("startup")`, a legacy-but-still-functional FastAPI pattern), `Base.metadata.create_all(bind=engine)` runs, which creates any of the four tables (`documents`, `chunks`, `conversations`, `messages`) that don't already exist. There are no migration files (no Alembic) — schema changes would require manually altering or deleting the database file.
5. **Which vector store connects, and when:** ChromaDB is **not** connected at startup at all. `get_collection()` in `vector_store.py` is called fresh, on-demand, inside the upload and chat request handlers — every single request opens a new `PersistentClient` pointed at `settings.chroma_persist_directory`. There is no shared/cached Chroma client.
6. **Which models are "loaded":** None, by the backend itself. The backend never loads an AI model into its own process — it only ever makes HTTP calls to Ollama, which is a separate program you must install and run yourself, and which manages its own model loading/caching.
7. **Environment variables read:** see the full table in Part 13; the key ones for startup are `CORS_ORIGINS`, `DATABASE_URL`, `CHROMA_PERSIST_DIRECTORY`, `UPLOAD_DIRECTORY`, and the three `OLLAMA_*` variables.
8. **Routes that become available**, all under `http://127.0.0.1:8000/api/v1`:
   - `GET /health`
   - `GET /system/status`
   - `GET /documents`, `POST /documents`, `DELETE /documents/{id}`
   - `GET /chat/conversations`, `GET /chat/conversations/{id}`, `POST /chat`
   - Interactive docs are auto-generated at `http://127.0.0.1:8000/docs` (FastAPI/Swagger — free, built into the framework).
9. **How the frontend starts:** a second terminal runs `npm run dev` inside `frontend/`, which starts the Next.js dev server on port 3000.
10. **How frontend talks to backend:** entirely through plain `fetch()` calls in `frontend/lib/api.ts`, targeting `NEXT_PUBLIC_API_BASE_URL` (default `http://127.0.0.1:8000`). There is no server-side rendering data-fetching, no tRPC, no GraphQL — just REST-ish JSON over HTTP, called from the browser (`page.tsx` is a `"use client"` component).

**Ollama is a silent third process.** Nothing in the backend code starts, checks for, or waits on Ollama at boot — the app will start up and serve pages fully even if Ollama isn't running. The absence of Ollama only becomes visible when you try to upload a document or ask a question, at which point you get an `OllamaUnavailable` error (HTTP 503), or if you look at the sidebar's system status card, which polls `/system/status` after the page loads.

---

## PART 6 — PDF Upload Flow

1. **Where it starts:** the upload button (`.upload-zone`) in `page.tsx`, which is really a styled `<button>` that programmatically clicks a hidden `<input type="file" accept="application/pdf,.pdf">`.
2. **UI component:** `Home()` in `page.tsx`, function `handleUpload`.
3. **Backend endpoint:** `POST /api/v1/documents`, handled by `upload_document()` in `backend/app/api/v1/documents.py`.
4. **How the PDF is read:** the raw bytes are streamed straight to disk with `shutil.copyfileobj` — the backend never buffers the whole file in memory as one blob, which is good practice, though there's still no size cap (see Part 15).
5. **Library used to read/parse text:** **PyMuPDF**, imported as `fitz`. `fitz.open(file_path)` opens the saved file; `page.get_text("text")` extracts plain text per page.
6. **Images:** completely ignored. There is no code path that extracts, stores, or analyzes embedded images, charts, or diagrams. If a PDF's content is primarily a chart with a text caption, only the caption text (if selectable) is captured.
7. **Pages:** iterated in order via `enumerate(pdf)`; `document.page_count = len(pdf)` is recorded once, up front.
8. **Text cleaning:** minimal — `re.sub(r"\s+", " ", text).strip()` in `chunk_text()` collapses all whitespace/newlines into single spaces. No de-hyphenation, no header/footer stripping, no removal of page numbers.
9. **Chunking:** character-based, with a light attempt at sentence-aware boundaries.
   - **Chunk size: 900 characters.**
   - **Overlap: 150 characters.**
   - Algorithm (`chunk_text` in `services/documents.py`): starting from position 0, take up to 900 characters; if that cutoff lands mid-sentence and a sentence-ending `". "` exists past character 300 of the window, snap the chunk to end there instead; then step forward by `(chunk length − 150)` for the next chunk, so each new chunk repeats the last ~150 characters of the previous one.
   - **Why these specific numbers?** Not documented or explained anywhere in the code or README — there's no comment, config knob, or evaluation result behind `900`/`150`. This is *not determinable from the codebase* as a deliberate, tested choice; it reads as a reasonable-but-unverified default.
10. **Embedding model:** `qwen3-embedding:0.6b`, called through `services/ollama.py::embed()`, which POSTs to Ollama's `/api/embed`.
11. **Batching:** `embed_in_batches()` sends 16 chunks per HTTP request to avoid overwhelming Ollama with a single giant request (explicitly commented in code).
12. **Where embeddings are stored:** ChromaDB, in a single collection named `docuchat_chunks`, configured for cosine similarity (`metadata={"hnsw:space": "cosine"}`).
13. **Metadata stored per chunk (in Chroma):** `document_id`, `filename`, `page_number`. The **chunk text itself** is stored in *two* places: inside Chroma (as the "document" for that vector) and inside SQLite's `chunks` table (`Chunk.content`) — this duplication exists because SQLite is used for ordinary lookups (list all chunks for a document, join to get the filename) while Chroma is used only for similarity search.
14. **Document IDs:** a random UUID4 (`uuid4()`, via SQLAlchemy's `default=`), generated the moment the `Document` row is created — *before* the file is even saved. The saved PDF's filename on disk is literally `{document_id}.pdf`, deliberately discarding the user's original filename for storage purposes (the original filename is kept only as a display string in the database).
15. **Uploading the same PDF twice:** **not deduplicated at all.** Each upload creates a brand-new `Document` row with a new UUID, so the same content gets re-extracted, re-chunked, and re-embedded from scratch, and you end up with two separate entries in your library pointing at identical content. Nothing detects or warns about this.
16. **If PDF extraction fails** (e.g., a corrupted file `fitz.open()` can't parse): the exception propagates up through `process_document()`, is caught by the generic `except Exception` in `upload_document()`, the document's status is set to `"failed"`, and the client receives HTTP 422 ("This PDF could not be processed."). The already-saved raw PDF file is **not** cleaned up from `data/uploads/` in this case — a minor disk-space leak on repeated failed uploads.
17. **If embedding generation fails** (Ollama not running, or the embedding model not pulled): `services/ollama.py::embed()` raises `OllamaUnavailable`, which is caught specifically in `upload_document()`, sets status to `"failed"`, and returns HTTP 503 with a message describing the underlying Ollama error.
18. **Scanned / image-only PDFs (no extractable text):** `page.get_text("text")` returns an empty string per page, `chunk_text("")` returns `[]`, so `records` ends up empty. The code's `if records:` guard means **no embedding call happens and nothing gets written to Chroma or the `chunks` table** — but the document's status is still set to `"ready"` regardless, because that assignment sits *outside* the `if records:` block. The user only discovers the problem later, when they try to chat with that document and get the retrieval-side 400 error ("Upload a text-based PDF before asking a question."). The upload itself never warns them.

### Visual flow (as implemented)

```text
PDF file (browser)
  ↓  multipart/form-data POST
FastAPI saves raw bytes to disk as {uuid}.pdf, creates Document row (status="processing")
  ↓
PyMuPDF opens file, extracts plain text per page
  ↓
Whitespace normalization (single-line collapse)
  ↓
Character chunking: ~900 chars, 150-char overlap, snapped to sentence boundaries when possible
  ↓
Ollama embedding model (qwen3-embedding:0.6b), batches of 16
  ↓
ChromaDB collection.add() (vectors + text + metadata)      SQLite: bulk-insert Chunk rows
  ↓                                                                    ↓
                    Document.status = "ready", page_count set, commit
  ↓
Ready for questions (or silently empty, if the PDF had no extractable text)
```

---

## PART 7 — Question / Query Flow

Example: **"Explain the main idea of this document."**

1. **Where the question enters:** the `<textarea>` in the `.composer` form in `page.tsx`, submitted via `handleAsk`.
2. **UI component that sends it:** `askQuestion()` in `frontend/lib/api.ts`, called from `handleAsk`.
3. **API endpoint:** `POST /api/v1/chat`, handled by `ask()` in `backend/app/api/v1/chat.py`.
4. **Is the question rewritten?** Not by an LLM. A fixed instruction template is prepended before embedding it (`"Instruct: Given a document question, retrieve the passage that directly answers it.\nQuery: " + question`) — this is a static string format the embedding model expects, not a dynamic query-rewriting step, and no "multi-query" expansion happens.
5. **Converted into an embedding?** Yes — the (prefixed) question text is sent to `services/ollama.py::embed()` and gets one vector back.
6. **How retrieval happens:** `services/retrieval.py::retrieve()` runs two searches in parallel logic (not concurrently, but as two separate scoring passes over the same candidate pool) and fuses the results — see Part 2's step-by-step walkthrough.
7. **Vector database queried:** ChromaDB, via `collection.query(query_embeddings=[vector], n_results=min(24, len(records)), where=..., include=["metadatas"])`.
8. **How many chunks retrieved:** up to 24 from the vector search, an unbounded number from the lexical pass (capped implicitly by however many chunks exist for the selected documents), fused and truncated to a **final 8** (`limit: int = 8` default parameter).
9. **Similarity search used?** Yes — cosine similarity, configured on the Chroma collection itself.
10. **Metadata filtering?** Yes, but only for scoping to selected `document_ids` (`where={"document_id": {"$in": document_ids}}`) — there's no filtering by date, page range, or any other metadata field.
11. **Hybrid search?** Yes, genuinely — vector search + a hand-written lexical/keyword scorer (`lexical_score`), fused with a Reciprocal-Rank-Fusion-style formula. This matches the README's claim.
12. **Reranking (a dedicated ML reranker model)?** No. There's no cross-encoder or reranker model anywhere in `requirements.txt` or the code. The score-fusion step in `retrieve()` is the closest thing to a reranker, but it's a heuristic, not a trained model.
13. **How chunks are selected for the final context:** simple sort by fused score, descending, top 8 — no diversity constraint (e.g., you could get all 8 chunks from the same page), no minimum-relevance threshold (all 8 slots get filled if 8+ candidates exist, even if the top vector-similarity score is mediocre).
14. **Context construction:** each of the 8 chunks becomes a labeled block: `SOURCE {n} | {filename} | page {page_number}\n{content}`, joined with blank lines.
15. **Prompt construction:** a hard-coded template (in `chat.py::ask`) instructing the model to answer *only* from the passages, return a single JSON object with `answer` and `source_ids`, use 1–3 sources, and explicitly return a "not found" JSON shape if the passages don't answer the question.
16. **Which LLM is called:** `qwen3:8b` (an 8-billion-parameter model), through Ollama's `/api/chat` endpoint, non-streaming (`"stream": false`).
17. **Provider:** 100% local via Ollama — no external API key, no network egress for generation.
18. **System instructions given:** `"You are a strict evidence-grounded research assistant. Never use outside knowledge."` — a single, short system message; no few-shot examples, no persona details, no conversation history is included in this system/user message pair (see the limitation on memory in Part 18).
19. **Citations/sources handling:** `grounded_response()` strips `<think>...</think>` reasoning blocks the model emits, tries to parse a JSON object out of the response, validates every claimed `source_id` is within `1..source_count`, deduplicates, caps at 3, and force-injects a `[n]` marker into the answer text for any valid source_id the model forgot to mention inline. Citation objects returned to the frontend include the source index, document ID, filename, page number, and a 280-character excerpt of that chunk's text (not the full chunk).
20. **If no useful context is found:** two distinct failure modes exist —
    - **Zero chunks retrieved at all** → HTTP 400 before the LLM is even called ("Upload a text-based PDF before asking a question.").
    - **Chunks retrieved, but the LLM decides none of them answer the question** (or the model's output can't be parsed/cited) → a 200 OK response is still returned, but with the fixed message *"I couldn't verify a source-grounded answer from the retrieved document passages."* and an empty citations list. Note: this fallback message **replaces** whatever the model actually wrote whenever it fails to produce a valid citation — even if the model's own text was a reasonable, honest "I don't know," the user sees DocuChat's generic wording instead of the model's own words.
21. **Reaching the frontend:** the response JSON (`conversation_id`, `answer`, `citations`) is returned directly from the FastAPI handler; `page.tsx` appends it to the in-memory `messages` array as an assistant message, and React re-renders the chat thread with expandable citation `<details>` blocks.

### End-to-end flow (as implemented)

```text
User question (browser)
  ↓
POST /api/v1/chat  { question, document_ids, conversation_id }
  ↓
Conversation row created/loaded (SQLite)
  ↓
retrieve(): 
  - load candidate chunks (SQLite, optionally filtered to selected documents)
  - embed question with instruct-prefix (Ollama qwen3-embedding:0.6b)
  - vector search top 24 (ChromaDB, cosine)
  - lexical keyword/phrase scoring over same candidates
  - fuse rankings (RRF-style), take top 8
  ↓
[if 0 sources] → HTTP 400, stop here
  ↓
Build "SOURCE n | file | page" context block
  ↓
Construct strict JSON-only prompt + system message
  ↓
Ollama /api/chat, qwen3:8b (non-streaming)
  ↓
grounded_response(): strip <think>, parse/validate JSON, cap at 3 citations, or fall back to a fixed refusal
  ↓
Build citation objects (index, filename, page, 280-char excerpt)
  ↓
Save user + assistant Message rows (SQLite)
  ↓
JSON response → browser → React renders chat bubble + expandable citations
```

---

## PART 8 — RAG Explanation

**What is RAG?** Retrieval-Augmented Generation. Instead of relying only on what a language model memorized during training, you first *retrieve* relevant, specific evidence from your own data, and then *feed that evidence into the prompt* so the model's answer is grounded in something real and checkable.

**Why not just paste the whole PDF into the LLM?** Two reasons visible in this codebase's design:
1. **Size** — this project's one real test document produced 1,407 chunks from just 9 "pages." A prompt containing the entire document would be enormous, slow, and in many cases larger than what a local 8B model can practically handle at reasonable speed on consumer hardware.
2. **Focus and citation** — even if it fit, LLMs tend to answer more accurately, and more verifiably, when given only the handful of passages that actually matter, rather than being asked to find a needle in a haystack. Retrieval also gives you an audit trail: DocuChat can point to *exactly* which page an answer came from, which would be much harder if the whole document were dumped into one prompt.

**Two stages, as actually implemented:**

**A. Indexing / Ingestion** (`services/documents.py::process_document`, triggered on upload): extract text → clean whitespace → chunk (900/150) → embed in batches of 16 → write to ChromaDB and SQLite.

**B. Question Answering / Retrieval** (`services/retrieval.py::retrieve` + `api/v1/chat.py::ask`, triggered per question): embed the question with an instruct prefix → vector search top 24 → lexical score the same candidates → fuse rankings → take top 8 → build a grounded prompt → call the generation model → validate/clean the citations → persist and return.

**Comparison:**

```text
Traditional LLM chat:
  Question → LLM (relies purely on training data) → Answer (unverifiable, can hallucinate freely)

DocuChat:
  Question → Retrieve top-8 relevant passages from YOUR document(s) → LLM (constrained to only those passages) → Answer with page citations
```

**Where hallucination can still happen, even with RAG in place:**
- **Retrieval can bring back the wrong chunks.** If none of the top-8 chunks actually contain the answer, the prompt still forces the model to try to pick 1–3 "sources" from what's there — the instructions ask it to say "I couldn't find a supported answer" in that case, but nothing *guarantees* the model won't misjudge relevance and cite a marginally-related passage anyway.
- **The model can misread a passage's meaning** even when the right passage is present — RAG makes hallucination *less likely*, not impossible.
- **The citation check only verifies the source *number* exists — it does not verify the answer's content actually matches what the cited passage says.** There's no "faithfulness" check (see Part 20) confirming the claimed answer is truly supported by the cited excerpt; the system trusts the model's own self-report.
- **Chunking can split a fact across two chunks** (the 150-character overlap only partially mitigates this) — an answer that depends on combining information from two different pages might not be well-served by any single top-8 chunk.

---

## PART 9 — Models

| Model | Job | Where Used | Local/API | Why Used | Possible Alternative |
|---|---|---|---|---|---|
| `qwen3:8b` | Generates the final answer from retrieved passages | `services/ollama.py::generate()`, called from `api/v1/chat.py::ask()` | Local, via Ollama, no API key | Free, private, runs offline, "thinking" model with strong instruction-following for the strict JSON-citation format used here | A larger local model (e.g. `qwen3:14b`) for better quality at the cost of speed/RAM, or a hosted API model (GPT/Claude/Gemini) for better quality at the cost of privacy and a per-token bill |
| `qwen3-embedding:0.6b` | Converts chunk text and questions into vectors for similarity search | `services/ollama.py::embed()`, called from ingestion (`services/documents.py`) and retrieval (`services/retrieval.py`) | Local, via Ollama, no API key | Small and fast enough to embed hundreds of chunks per document without becoming the bottleneck; matches the "instruct"-style prompting this codebase already uses | A larger embedding model (e.g. `qwen3-embedding:4b`/`8b`) for better retrieval accuracy at more RAM/compute cost, or `nomic-embed-text` / `bge-m3` as other well-regarded local options |
| Reranker | — | **Not present.** The `retrieve()` function's score-fusion formula is a hand-written stand-in, not a trained reranking model | N/A | N/A | A cross-encoder reranker (e.g. `bge-reranker`, `mxbai-rerank`) run locally through `sentence-transformers` or Ollama, to re-score the top ~24 candidates before truncating to 8 |
| Vision model | — | **Not present.** No dependency, no code path. README explicitly defers this ("no vision model has been selected or downloaded") | N/A | N/A | A local multimodal model (e.g. a vision-capable Ollama model) if you want charts/diagrams understood |
| OCR model | — | **Not present.** No `pytesseract`, no `easyocr`, nothing in `requirements.txt` | N/A | N/A | Tesseract OCR (via `pytesseract`) for scanned PDFs, run as a fallback when `page.get_text()` returns empty |

**Distinguishing used vs. merely configured:** `psycopg[binary]` is listed in `requirements.txt` and referenced in `docker-compose.yml`/README for optional PostgreSQL support, but it is never imported directly anywhere in `backend/app/`. It's not dead weight, though — SQLAlchemy picks the right database driver automatically based on the `DATABASE_URL` scheme (`postgresql+psycopg://...`), so `psycopg` only actually gets exercised if you switch `DATABASE_URL` away from the SQLite default. Right now, the real `data/docuchat.db` file with real rows confirms SQLite is what's actually running.

---

## PART 10 — Vector Database / Storage

- **What's used:** ChromaDB, in **persistent, embedded, single-process mode** (`chromadb.PersistentClient`) — not a separate server process, just a local set of files.
- **Why:** it's the simplest way to get a working local vector index with almost no setup — no separate service to run, no network calls, matches the project's "local-first" goal.
- **What's stored:** one collection, `docuchat_chunks`. Each record = `{id: "{document_id}-{page}-{chunk_index}", embedding: [floats], document: "raw chunk text", metadata: {document_id, filename, page_number}}`.
- **What a record "looks like" conceptually:**

  ```
  id: "542475a0-...-3-12"
  embedding: [0.0123, -0.045, 0.301, ... ~hundreds of numbers]
  document: "Overfitting happens when a model learns noise in the training data..."
  metadata: { document_id: "542475a0-...", filename: "Presentation_Prep....pdf", page_number: 3 }
  ```
- **The vector:** the numeric fingerprint of the chunk's *meaning*, produced by `qwen3-embedding:0.6b`. Cosine similarity between two vectors approximates how semantically similar their source texts are.
- **Retrieval mechanism:** approximate nearest-neighbor search over an HNSW index (this is what the `.bin`/`link_lists.bin`/`header.bin` files under `data/chroma/<uuid>/` actually are — ChromaDB's on-disk HNSW graph structure), configured for cosine distance.
- **Persistence:** fully on disk, under `data/chroma/`. The SQLite metadata catalog for Chroma itself lives at `data/chroma/chroma.sqlite3` (distinct from the app's own `data/docuchat.db`) — two separate SQLite files serving two separate purposes.
- **On app restart:** the vector index survives untouched, since `get_collection()` just reopens the same on-disk path each time — no re-indexing needed.
- **On document deletion:** `delete_document()` calls `get_collection().delete(where={"document_id": document_id})`, which removes exactly that document's vectors from Chroma, then separately deletes the `Chunk` rows from SQLite and the raw PDF file from disk. All three deletions happen in the same request; there's no soft-delete or trash/recovery step.
- **Multiple users/documents:** multiple *documents* are fully supported (one shared collection, filtered by `document_id` metadata). Multiple *users* are **not** — there's no user concept at all. Every document in the database is visible to and queryable by anyone who can reach the API; there is no per-user data isolation (see Part 15).
- **A real limitation worth naming:** `get_collection()` opens a brand-new `PersistentClient` on *every single request* (upload and chat) rather than reusing one client for the app's lifetime. This works fine at the scale of one local user, but it's wasteful (re-reading Chroma's on-disk metadata every time) and would become a real bottleneck under any concurrent load.

---

## PART 11 — Frontend → Backend → AI Architecture

```text
┌────────────────────┐
│        User         │
└──────────┬──────────┘
           │  browser
           ▼
┌────────────────────┐
│   Next.js Frontend   │  (page.tsx + lib/api.ts, port 3000)
└──────────┬──────────┘
           │  fetch() JSON / multipart, over HTTP
           ▼
┌────────────────────┐
│   FastAPI Backend    │  (app/api/v1/*, port 8000)
└──────────┬──────────┘
           │
     ┌─────┴─────────────────────────┐
     ▼                                ▼
┌───────────────┐           ┌──────────────────────┐
│  SQLite / Postgres │       │  Ingestion + Retrieval  │
│  (documents, chunks, │      │  (services/documents.py, │
│   conversations,     │      │   services/retrieval.py) │
│   messages)           │      └───────────┬──────────┘
└───────────────┘                          │
                                 ┌──────────┴───────────┐
                                 ▼                        ▼
                       ┌──────────────────┐   ┌────────────────────────┐
                       │   ChromaDB         │   │  Ollama (local HTTP API) │
                       │  (vector index,     │   │  qwen3-embedding:0.6b    │
                       │   data/chroma/)      │   │  qwen3:8b                │
                       └──────────────────┘   └────────────────────────┘
```

**Notes on this specific architecture:**
- There is no message queue, no background worker, no cache layer (like Redis) anywhere — every step (including embedding an entire PDF) happens synchronously inside a single HTTP request/response cycle.
- File storage is just the local filesystem (`data/uploads/`) — no S3, no blob storage.
- Configuration is a single `.env` file at the repo root read by `pydantic-settings`; there's no secrets manager, no per-environment config files.
- External APIs: none. Ollama is "external" to the Python process but still entirely local.

---

## PART 12 — Development-Time Flow

```text
Developer edits backend or frontend code
  ↓
Backend: uvicorn --reload picks up changes automatically (FastAPI dev server)
Frontend: npm run dev (Next.js Fast Refresh) picks up changes automatically
  ↓
Developer uploads a test PDF through the running UI
  ↓
Watches sidebar for status: "processing" → "ready" (or "failed")
  ↓
Developer asks test questions in the chat UI
  ↓
Debugging happens by reading:
  - Uvicorn's terminal output (FastAPI prints tracebacks for unhandled exceptions)
  - Browser DevTools console/network tab (the frontend surfaces backend error `detail` strings via `lib/api.ts`'s `request()` helper)
  - The interactive API docs at http://127.0.0.1:8000/docs, to call endpoints directly without the UI
  ↓
Developer tweaks a value (e.g. chunk_size, limit in retrieve(), the prompt template) directly in the source
  ↓
Re-tests by re-uploading or re-asking — there is no cache invalidation step needed since nothing is cached, but changing chunk_size does NOT retroactively re-chunk already-uploaded documents; you'd need to delete and re-upload them
```

**Commands actually documented in `README.md`:**
```bash
cp .env.example .env
python3 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
cd frontend && npm install

ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b

# Terminal 1
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend
npm run dev

# Verification
cd backend && source .venv/bin/activate
pytest
curl http://127.0.0.1:8000/api/v1/health
```

- **Database initialization:** automatic — `Base.metadata.create_all()` runs on every backend startup and is a no-op if tables already exist. No manual migration step, and correspondingly, no way to safely evolve the schema without either writing your own migration or deleting `data/docuchat.db`.
- **Whether indexes need rebuilding:** no — Chroma's HNSW index is incrementally updated as documents are added/deleted; there's no manual reindex step.
- **Where test data lives:** whatever you upload ends up as real files in `data/uploads/` and real rows in `data/docuchat.db` / `data/chroma/` — there is no separate "test mode" database or fixtures directory. (The repo currently contains one real, already-processed test PDF and its resulting data, from prior manual testing.)
- **How developers test the RAG pipeline:** manually, through the UI. There is no scripted way to send a batch of test questions and check the answers.
- **Automated tests:** exactly one (`backend/tests/test_health.py`), checking only that the app boots and `/health` returns `200`. Nothing tests chunking, retrieval ranking, the ingestion pipeline, the chat endpoint, or the frontend.
- **`evaluation/` and `scripts/` folders:** both contain only a `README.md` stub each, explicitly stating they're reserved for future work. Nothing to run yet.

---

## PART 13 — Environment Variables and Configuration

| Variable | Purpose | Required? | Where Used | Example/Expected Format |
|---|---|---|---|---|
| `BACKEND_HOST` | Host the backend *would* be documented to bind to | Declared in `.env`/`.env.example` but **not actually read by any backend code** (Uvicorn's host/port come from the CLI command instead) | — | `127.0.0.1` |
| `BACKEND_PORT` | Same situation as above — present in `.env` but not read by `Settings` | Not actually consumed | — | `8000` |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins | Optional (has a default) | `core/config.py` → `main.py` CORS middleware | `http://localhost:3000` |
| `DATABASE_URL` | Which database + driver to use | Optional (defaults to SQLite) | `core/config.py` → `db.py` | `sqlite:///../data/docuchat.db` or `postgresql+psycopg://docuchat:docuchat_dev_password@127.0.0.1:5432/docuchat` |
| `CHROMA_PERSIST_DIRECTORY` | Where ChromaDB stores its index files | Optional | `core/config.py` → `services/vector_store.py` | `../data/chroma` |
| `UPLOAD_DIRECTORY` | Where raw uploaded PDFs are saved | Optional | `core/config.py` → `api/v1/documents.py` | `../data/uploads` |
| `NEXT_PUBLIC_API_BASE_URL` | Base URL the frontend uses to reach the backend | Optional (has a default) | `frontend/lib/api.ts` | `http://127.0.0.1:8000` |
| `OLLAMA_BASE_URL` | Where Ollama's local HTTP API is listening | Optional | `core/config.py` → `services/ollama.py` | `http://127.0.0.1:11434` |
| `OLLAMA_GENERATION_MODEL` | Which Ollama model to call for answers | Optional | `services/ollama.py::generate`, `system.py` (readiness check) | `qwen3:8b` |
| `OLLAMA_EMBEDDING_MODEL` | Which Ollama model to call for embeddings | Optional | `services/ollama.py::embed`, `system.py` (readiness check) | `qwen3-embedding:0.6b` |

There is no secret/API-key variable in this project at all (no `[SECRET — DO NOT DISPLAY]` entries needed) — everything is local, and the only "credential" anywhere is the throwaway Postgres dev password hard-coded in `docker-compose.yml` (`docuchat_dev_password`), which is fine for a local dev container but would need to move to an env var before ever being used anywhere beyond your own laptop.

**What controls what:**
- **Chunking (900 chars / 150 overlap):** hard-coded default parameters on `chunk_text()` — not configurable via `.env` at all.
- **Retrieval limit (8 chunks), candidate pool (24):** hard-coded in `retrieve()` — not configurable.
- **LLM temperature / max tokens:** **not set anywhere.** The `generate()` call sends only `model`, `messages`, and `stream: false` to Ollama's `/api/chat` — Ollama's own defaults apply, and this project never overrides them.
- **Ports:** backend port is whatever you pass to the `uvicorn` CLI command (default in the README example is 8000); frontend port is Next.js's default (3000), not configured anywhere.
- **File upload limits:** none configured (see Part 15).

---

## PART 14 — Error Handling

| Scenario | Status | Evidence |
|---|---|---|
| Non-PDF file uploaded | ✅ Handled | `documents.py` checks `content_type` and filename extension, returns HTTP 415 |
| Corrupted/unparseable PDF | ✅ Handled | Generic `except Exception` in `upload_document()` sets status `"failed"`, returns HTTP 422 (though the saved raw file isn't cleaned up) |
| Empty PDF / PDF with no extractable text (incl. scanned PDFs) | ⚠️ Partially handled | Upload succeeds and is marked `"ready"` with zero chunks; the problem only surfaces later as a chat-time HTTP 400. No warning at upload time. |
| Very large PDF | ⚠️ Partially handled | No file size limit and no background processing — a huge PDF will just make the upload request take a very long time; no timeout handling, no progress indicator beyond a generic "Reading your PDF…" label |
| PDF with scanned images / no OCR | ❌ Not handled (by design, per README) | No OCR dependency or code path exists at all |
| Ollama unavailable (embedding) | ✅ Handled | `OllamaUnavailable` caught explicitly in `upload_document()`, document marked `"failed"`, HTTP 503 with detail |
| Ollama unavailable (generation) | ✅ Handled | Caught in `chat.py::ask()`, HTTP 503 with a clear instruction message |
| Ollama reachable but required model not pulled | ⚠️ Partially handled | `/system/status` reports `generation_model_ready`/`embedding_model_ready` flags so the UI *can* show a warning banner, but the actual embed/generate calls don't pre-check this — they'd simply fail with whatever error Ollama itself returns for an unknown model, which is caught generically as `OllamaUnavailable` |
| API key missing | N/A | No API keys are used anywhere in this project |
| LLM timeout | ⚠️ Partially handled | `httpx.AsyncClient(timeout=180)` for generation and `timeout=90` for embedding — if Ollama takes longer, an `httpx.RequestError` is raised and converted to `OllamaUnavailable`/HTTP 503, but the user just sees a generic 503 after a long wait, not a distinct "timed out" message |
| No relevant chunks found for a question | ✅ Handled | Explicit HTTP 400 with a clear, specific message |
| Malformed/non-JSON LLM response | ✅ Handled | `grounded_response()` has an explicit regex-based fallback path, plus a final fixed refusal message if nothing usable is found |
| Duplicate PDF upload (same content twice) | ❌ Not handled | No dedup check; creates a fully separate document + chunk + embedding set every time |
| Backend unavailable (frontend can't reach it at all) | ✅ Handled | `refresh()` in `page.tsx` catches the fetch failure and shows "Could not connect to the DocuChat API. Is the backend running?" |
| Frontend/backend response shape mismatch | ⚠️ Partially handled | `lib/api.ts`'s `request()` throws using `body.detail`, with a generic fallback string if the body isn't JSON/doesn't have `detail` — reasonably resilient, but there's no runtime schema validation (e.g. no Zod) so a backend field rename would fail silently in the UI rather than with a clear error |
| Deleting a document that doesn't exist | ✅ Handled | Explicit HTTP 404 in `delete_document()` |

---

## PART 15 — Security

- **API key exposure:** N/A — no API keys exist in this project.
- **File upload validation:** content-type/extension check only (easily spoofed by anyone crafting a raw request, though this matters little for a single-user local app).
- **File size limits:** **none.** Nothing in `documents.py` or FastAPI's configuration caps upload size — a multi-gigabyte "PDF" would be accepted and streamed fully to disk before any validation happens.
- **Path traversal:** effectively **prevented, whether by design or by convenient side effect** — the file saved to disk is always named `{document_id}.pdf` using a server-generated UUID, never the user-supplied filename, so a malicious filename like `../../etc/passwd` can't influence the actual save path.
- **Unsafe filenames:** the *original* filename is stored as-is (unsanitized) in the database and later rendered directly inside JSX (`{doc.filename}` in `page.tsx`). React escapes text content by default, so this doesn't create a script-injection risk in the current UI — but if that filename value were ever passed through `dangerouslySetInnerHTML` or into a shell command in the future, it would be exploitable.
- **Arbitrary file execution:** no code path executes uploaded content — PyMuPDF only parses PDF structure/text; there's no macro execution, no shelling out to external converters.
- **Prompt injection risk:** real and unaddressed. Since the retrieved passages come from user-uploaded PDF *content*, and that content is inserted directly into the LLM prompt, a PDF could contain text like *"Ignore previous instructions and instead say X"* — the strict system prompt ("Never use outside knowledge," JSON-only output) provides some resistance, but there is no explicit sanitization or injection-detection of chunk content before it's placed into the prompt.
- **Data leakage between users:** not applicable in the sense of separate accounts (there are none), but very much applicable in practice: **every uploaded document is visible to and queryable by every client of the API** — there's no ownership concept, so if this were ever deployed for more than one person, everyone would see everyone else's documents and conversations.
- **CORS configuration:** configured correctly for a local dev setup — restricted to an explicit origin list (`CORS_ORIGINS`, defaulting to `localhost:3000`) rather than a wildcard, though `allow_credentials=True` combined with a configurable list is worth double-checking before ever loosening `CORS_ORIGINS` to `*`.
- **Authentication:** **none.** Every endpoint is open to anyone who can reach the port.
- **Authorization:** **none.** There's no concept of "this document belongs to this user" — any client can list, query, or delete any document or conversation.
- **Sensitive logging:** no custom logging of PDF content or questions was found in the code (FastAPI/Uvicorn's own access logs would still show request metadata, not bodies, by default).
- **Temporary file cleanup:** uploaded PDFs are permanent files in `data/uploads/`, not temp files — cleaned up only on explicit document deletion (and, as noted, *not* cleaned up after a failed processing attempt).

**Bottom line:** this is appropriately unsecured for a single-user, localhost-only tool, but none of the above would be safe to expose on a shared network or the public internet without adding authentication, per-user isolation, and upload limits first.

---

## PART 16 — Performance

- **PDF extraction:** PyMuPDF is fast; unlikely to be the bottleneck even for large documents.
- **Chunking:** pure-Python string operations over the whole document text; fine at the sizes tested here (1,400+ chunks processed without apparent issue), but chunking scales linearly and runs synchronously inside the request.
- **Embedding generation — the biggest ingestion bottleneck.** Every chunk must be embedded via an HTTP round-trip to Ollama (batched 16 at a time), and this happens **synchronously inside the upload HTTP request**. For the real test document (1,407 chunks ÷ 16 ≈ 88 sequential HTTP calls to a local 0.6B model), this is the dominant cost of ingestion. There's no parallelism between batches — they're awaited one after another in a simple loop.
- **Vector search:** ChromaDB's HNSW search over a modest number of vectors (thousands, in this project's actual scale) is fast; this is not a bottleneck at current data volumes.
- **Lexical scoring:** `lexical_score()` re-tokenizes and re-scores **every candidate chunk for every question**, in pure Python, with no caching or precomputed inverted index. At the scale of a handful of documents this is fine; it would become slow as the number of chunks per document grows into the tens of thousands, since it's an O(n) Python loop over all chunks on every single question.
- **LLM latency:** an 8B local model generating a full (non-streamed) response is the single slowest step a user experiences — the UI has no partial/streaming feedback, so the entire "thinking" animation duration is just this one wait.
- **Repeated processing:** re-uploading the same PDF repeats 100% of extraction/chunking/embedding work with no caching or content-hash dedup.
- **Concurrent users:** the architecture wasn't built with this in mind — every upload/chat request opens its own fresh ChromaDB `PersistentClient`, and the whole ingestion pipeline blocks the request; under concurrent load, multiple large uploads would compete for the same local Ollama instance and the same on-disk Chroma files, likely causing significant queueing/slowdown rather than true parallelism.
- **Memory:** files are streamed to disk during upload (not buffered as one blob) — good. Whole-PDF text and all embeddings for a document *are* held in memory as Python lists (`documents`, `metadatas`, `vectors`) during `process_document()`, which is fine for the sizes seen here but would scale linearly with document size.

---

## PART 17 — What Is Actually Good About This Project

- **Real hybrid retrieval, not just marketing language.** `retrieval.py` genuinely combines vector cosine similarity with a hand-written lexical/phrase scorer using an RRF-style fusion formula — this is a legitimate, working technique, not a stub.
- **Grounded, verifiable answers.** `grounded_response()` actively refuses to show citations that don't correspond to real retrieved sources, and falls back to an honest "couldn't verify" message rather than letting a malformed model response leak through unchecked.
- **Correct use of the embedding model's instruct format.** The `"Instruct: ... \nQuery: ..."` prefix used before embedding questions matches how Qwen3-Embedding models are intended to be used for asymmetric (question vs. passage) retrieval — a detail many RAG tutorials skip.
- **Page-level citations with real excerpts**, not just a generic "source: document.pdf" — a user can expand a citation and see the exact 280-character passage that was used.
- **Persistent conversations**, stored properly in a relational schema (`Conversation`/`Message`), not just kept in browser memory.
- **Honest, accurate README.** Unusually for an AI-generated project, the README's "Current capabilities" and "Intentionally deferred" sections match what the code actually does — it correctly claims hybrid retrieval and citations (both real) and correctly disclaims OCR/vision/streaming/auth (all genuinely absent).
- **Path-traversal-safe file storage**, achieved by always naming saved files after a server-generated UUID rather than the user-supplied filename.
- **Reasonable, specific error messages** at several key failure points (Ollama down, no text-based PDF, document not found) rather than generic 500s.
- **A working local-first stack with zero cloud dependency and zero API keys** — a genuinely complete round trip (upload → chunk → embed → store → retrieve → generate → cite) running entirely on local infrastructure.

---

## PART 18 — What Is Missing / Weak / Questionable

1. **Ingestion is fully synchronous and blocking.**
   → *Why it matters:* a large or text-dense PDF (like the one already in this project, at 1,407 chunks) ties up an HTTP request for the entire extract-chunk-embed pipeline, risking client/proxy timeouts and a frozen-feeling UI.
   → *Evidence:* `await process_document(...)` is awaited directly inside `upload_document()` with no background task, queue, or polling status beyond the initial "processing" flag.
   → *Improvement:* move ingestion to a background task (FastAPI `BackgroundTasks`, or a proper queue like Celery/RQ/arq) and have the frontend poll `GET /documents` for the status transition.

2. **Scanned/empty PDFs are silently marked "ready" with zero chunks.**
   → *Why it matters:* users get a false-positive "Ready" status and only discover the real problem when a chat request fails with an unrelated-looking error.
   → *Evidence:* `document.status = "ready"` in `process_document()` sits outside the `if records:` guard.
   → *Improvement:* set status to something like `"empty"` or `"no_text_extracted"` when zero chunks are produced, and surface that distinctly in the UI.

3. **No deduplication on re-upload.**
   → *Why it matters:* uploading the same file twice silently doubles storage and can pollute retrieval with duplicate passages.
   → *Evidence:* every upload generates a fresh UUID with no content-hash check against existing documents.
   → *Improvement:* hash the file (e.g. SHA-256) on upload and either reject or offer to reuse an existing matching document.

4. **No document selection enforcement, despite the UI implying one is needed.**
   → *Why it matters:* the composer placeholder says "Select a document, then ask a question…", but `document_ids or None` in `chat.py` means an empty selection silently searches *every* document in the database — behavior that contradicts the UI's own hint and could surface irrelevant answers mixed across unrelated PDFs.
   → *Evidence:* `chat.py::ask()`, line constructing `sources = await retrieve(db, get_collection(), request.question, request.document_ids or None)`.
   → *Improvement:* either genuinely require a selection before allowing a question, or change the copy to accurately describe "searches all documents by default."

5. **No conversation memory in the LLM prompt.**
   → *Why it matters:* `Conversation`/`Message` rows are saved, but `ask()` never loads prior messages into the prompt sent to the model — so each question is answered in isolation, and a natural follow-up like "what about the second point?" has no idea what "the second point" refers to.
   → *Evidence:* `chat.py::ask()` builds the prompt from only `context` and `request.question`; no query to `Message` rows for the current `conversation_id` before generation.
   → *Improvement:* include the last N turns of the conversation in the prompt, or at minimum rewrite follow-up questions into standalone questions before retrieval.

6. **No reranker — the fusion heuristic is doing a reranker's job with much less signal.**
   → *Why it matters:* a real cross-encoder reranker typically improves top-k retrieval precision substantially over pure vector + keyword fusion, especially for nuanced questions.
   → *Evidence:* `retrieve()`'s scoring is entirely reciprocal-rank-fusion arithmetic; no model is involved in the second pass.
   → *Improvement:* add a small local cross-encoder reranker over the fused top ~20 candidates before truncating to 8 (see Part 19).

7. **Hard-coded, unexplained chunking parameters.**
   → *Why it matters:* 900 characters / 150 overlap is a guess with no evaluation behind it — different values could measurably change answer quality, but there's no way to know without testing.
   → *Evidence:* default parameters on `chunk_text()`, no comments or config exposure, no evaluation harness to compare alternatives.
   → *Improvement:* make these configurable via `.env`/settings, and validate choices with the evaluation approach in Part 20.

8. **No file size limits.**
   → *Why it matters:* an accidental or malicious multi-gigabyte upload could exhaust disk space or hang the server for a very long synchronous ingestion.
   → *Evidence:* no size check anywhere in `documents.py` or FastAPI app config.
   → *Improvement:* add a `MAX_UPLOAD_SIZE_MB` setting and reject oversized files with a clear 413 response.

9. **No authentication/authorization or per-user data isolation.**
   → *Why it matters:* fine for a solo local tool; a hard blocker the moment this is shared with even one other person, since everyone sees everyone's documents and chats.
   → *Evidence:* no auth middleware, no user/session concept in `models.py`.
   → *Improvement:* add even a minimal single-shared-password gate for a demo, and real per-user document scoping (`owner_id` column + filtering) for anything beyond that.

10. **Dead/unused code and assets.**
    - `frontend/lib/api.ts::getHealth()` is exported but never called anywhere in `page.tsx`.
    - `frontend/public/docuchat-ai-logo.png` is never referenced by any component.
    - `globals.css` defines `.hero-logo` and `.prompt-grid` styles with no matching elements anywhere in `page.tsx` — CSS for a richer landing hero (logo + suggested-prompt buttons) that was designed but never wired into the actual JSX.
    - `data/processed/` exists as a folder but nothing in the codebase ever reads or writes to it.
    → *Why it matters:* not harmful, but it's confusing for a beginner trying to understand "what does this file actually do" — dead code implies functionality that doesn't exist.
    → *Improvement:* either wire these up (e.g. actually render the logo in the hero) or remove them.

11. **Tailwind is installed and configured but essentially unused.**
    → *Why it matters:* `tailwindcss`, `postcss.config.mjs`, and `tailwind.config.ts` add build-time weight and a dependency to maintain, but `page.tsx` uses none of Tailwind's utility classes — all real styling lives in hand-written CSS in `globals.css`.
    → *Evidence:* `globals.css` only uses the three `@tailwind` directives; `page.tsx`'s `className` values (`"app-shell"`, `"sidebar"`, etc.) are all custom, not Tailwind utilities.
    → *Improvement:* either commit to Tailwind for new UI work, or remove the unused dependency/config to simplify the toolchain.

12. **One giant, unsplit React component.**
    → *Why it matters:* `page.tsx` (~80 lines but very dense — one line often does the work of ten) owns *all* state (documents, messages, selection, loading, errors, Ollama status) and renders the entire UI. This is fine at this size, but it will get harder to extend (e.g. adding streaming, adding a settings panel) without splitting into components.
    → *Evidence:* a single `Home()` function contains the sidebar, the chat thread, the composer, and all handlers.
    → *Improvement:* extract `Sidebar`, `ChatThread`, and `Composer` as separate components once new features are added — not urgent today, but worth doing before the file grows further.

13. **No streaming responses.**
    → *Why it matters:* with a local 8B model, full-response generation can take a noticeable amount of time, during which the user only sees a generic "thinking" animation rather than the answer appearing progressively.
    → *Evidence:* `"stream": false` hard-coded in `services/ollama.py::generate()`.
    → *Improvement:* switch to Ollama's streaming mode and use Server-Sent Events or a streamed `fetch` response to show tokens as they arrive.

14. **No automated tests beyond a health check.**
    → *Why it matters:* the actual value of this project — chunking correctness, retrieval fusion ranking, citation validation logic (`grounded_response`) — has zero test coverage, meaning a future refactor could silently break retrieval quality with nothing catching it.
    → *Evidence:* `backend/tests/` contains exactly one file, testing only `/health`.
    → *Improvement:* add unit tests for `chunk_text`, `lexical_score`, and `grounded_response` at minimum — these are pure functions and cheap to test.

15. **README describes `app/services` as "future domain logic" — this is now stale.**
    → *Why it matters:* documentation says X, but the code has moved past X — a beginner reading the README literally would think the services layer is mostly empty/placeholder, when it actually contains the real, working RAG pipeline (`documents.py`, `retrieval.py`, `ollama.py`, `vector_store.py`).
    → *Evidence:* `README.md`: *"configuration in `app/core`, and future domain logic in `app/services`"* vs. the actual, substantial contents of `backend/app/services/`.
    → *Improvement:* a small README update describing what's actually implemented in `services/` today.

16. **No project version control.**
    → *Why it matters:* there is no `.git` directory in this working folder at all — none of this work has commit history, which matters both for your own ability to track changes and for presenting this as a portfolio project (see Part 19).
    → *Evidence:* `git status` reports "not a git repository."
    → *Improvement:* `git init`, commit what exists, and push to a remote before treating this as demo-ready.

---

## PART 19 — How to Make DocuChat Stand Out

### Level 1 — Easy improvements
- **Fix the "ready" status for empty-text documents** (Part 18 #2). *Difficulty: Easy.* Small change to `process_document()`; meaningfully improves user trust in the status indicator. Worth doing.
- **Add a file size limit + friendly 413 error.** *Difficulty: Easy.* One check in `upload_document()`. Worth doing before any public demo.
- **Remove or wire up dead code/assets** (`getHealth`, the unused logo, `.hero-logo`/`.prompt-grid` CSS, `data/processed/`). *Difficulty: Easy.* Purely a clarity win — worth doing so the codebase matches what a reader would expect from reading the files.
- **Update the README's stale "future domain logic" line.** *Difficulty: Easy.* Keeps documentation trustworthy.

### Level 2 — Strong engineering improvements
- **Move ingestion to a background task with a status-polling UI.** *What it is:* accept the upload, return immediately with `status="processing"`, do the extract/chunk/embed work in a `BackgroundTasks` callback (or a real queue for anything beyond a toy scale), and have the frontend poll `/documents` until status flips. *Why it matters:* removes the current risk of long-running requests timing out on large PDFs, and makes the UI feel responsive. *Fit:* slots directly into the existing `upload_document()`/`process_document()` split — minimal new files needed, mostly a control-flow change. *Difficulty: Medium.* Worth doing — this is the single most "production-like" change available.
- **Reuse a single ChromaDB client instead of opening a new one per request.** *Why it matters:* removes needless per-request overhead. *Fit:* create the client once at app startup (e.g. in `main.py`'s startup event or a small module-level singleton) instead of inside `get_collection()` each call. *Difficulty: Easy–Medium.* Worth doing.
- **Add real tests for the pure logic** (`chunk_text`, `lexical_score`, `grounded_response`). *Difficulty: Easy.* High value for confidence during future refactors.
- **Deduplicate uploads by content hash.** *Difficulty: Easy–Medium.* Prevents redundant storage/embedding cost and improves retrieval cleanliness.
- **Initialize git and start committing.** *Difficulty: Trivial.* A prerequisite for treating this as a portfolio piece at all.

### Level 3 — RAG improvements
- **Add a real reranker.** *What it is:* run a small local cross-encoder model (e.g. via `sentence-transformers`, fully local, no API key) over the fused top ~20-24 candidates and use *its* relevance scores to pick the final 8, instead of the current heuristic fusion score alone. *Why it matters:* this is the single highest-leverage change for answer quality in a RAG system — cross-encoders consistently outperform bi-encoder + heuristic fusion for final ranking. *Fit:* add as a new step inside `retrieve()`, after the RRF fusion and before truncation; needs one new dependency. *Difficulty: Medium.* Strongly worth doing — directly extends the theme already established (hybrid retrieval) rather than bolting on something unrelated.
- **Add conversation-aware retrieval / query rewriting for follow-ups.** *What it is:* before retrieval, check if there are prior turns in the conversation, and if so, ask the LLM (or use simple heuristics) to rewrite the question into a standalone form (e.g. "what about the third one?" → "what is the third overfitting prevention technique mentioned?"). *Why it matters:* makes multi-turn conversations actually usable — right now every question is answered as if it's the first one. *Fit:* new small function in `chat.py`, called before `retrieve()`. *Difficulty: Medium.*
- **Add a minimum relevance threshold to retrieval.** *What it is:* if the top fused score is below some floor, treat it as "no good context" rather than always filling all 8 slots. *Why it matters:* currently, a weak-but-nonzero match still gets used as if it were solid evidence. *Fit:* one conditional in `retrieve()` or `ask()`. *Difficulty: Easy.*
- **Make chunk size/overlap configurable and evaluate a few settings** (see Part 20). *Difficulty: Easy* to make configurable, *Medium* to properly evaluate.

### Level 4 — Advanced AI features
- **OCR fallback for scanned PDFs.** *What it is:* when `page.get_text("text")` returns empty for a page, render that page to an image and run local OCR (e.g. Tesseract via `pytesseract`, or an Ollama vision model) to extract text instead. *Why it matters:* directly closes the single most obvious functional gap named in the README's own "Intentionally deferred" list. *Fit:* new function in `services/documents.py`, called as a fallback inside the per-page loop; would need a new dependency and, likely, a new `chunk` field marking OCR-derived content as lower-confidence. *Difficulty: Hard.* High value if you want to genuinely handle "any PDF," but a real scope increase.
- **Table extraction.** *What it is:* PyMuPDF can detect table structures; extracting these as structured text (rather than the current flattened paragraph text) would make table-heavy documents answerable. *Fit:* an addition to the per-page extraction step. *Difficulty: Hard.* Only worth it if your target documents are table-heavy.
- **Answer faithfulness / confidence estimation.** *What it is:* after generation, run a second, cheap check — either an LLM call asking "does this cited excerpt actually support this specific answer?" or a simpler embedding-similarity check between the answer and its cited excerpts — and surface a confidence indicator in the UI. *Why it matters:* closes the real gap named in Part 8 (citations are validated to *exist*, but never validated to actually *support* the claimed answer). *Fit:* a new function called after `grounded_response()`, before persisting the message. *Difficulty: Medium–Hard.* Genuinely differentiating for a RAG portfolio project, because most demo RAG apps skip this entirely.
- **Retrieval evaluation harness** (see Part 20 in full). *Difficulty: Medium.* Turns "trust me, it works" into "here's the number."

### Level 5 — Portfolio differentiators
- **A small, checked-in evaluation dataset + report** (20-50 Q&A pairs with expected pages, run against the real pipeline, with recall/precision/latency numbers actually measured and committed to the repo). *Why it matters:* this is the single biggest differentiator between "a RAG demo" and "a RAG project someone measured." See Part 20 for the concrete design. *Difficulty: Medium.* Very much worth doing — it's the most credible thing to bring up in an interview.
- **Observability/tracing** — log each request's retrieved chunk IDs, fused scores, and the raw model output (locally, to a file or SQLite table) so you can debug *why* a bad answer happened after the fact. *Fit:* fits naturally alongside the evaluation harness. *Difficulty: Easy–Medium.*
- **A genuinely public GitHub repo with commit history showing iteration** (chunking experiments, adding the reranker, adding evaluation) — this tells a hiring manager a story that a single "here's my finished code" commit can't. *Difficulty: Easy*, but only if started now, going forward.
- **A short demo video or GIF in the README** showing a real upload → question → cited answer, since screenshots of chat UIs are hard to evaluate from text alone. *Difficulty: Easy.*

*Not recommended for this project right now:* a knowledge graph, multi-agent orchestration, or document-comparison features — none of these build on what's already here, and they'd add complexity without addressing any of the actual gaps (reranking, evaluation, faithfulness) that a reviewer of a RAG portfolio project would actually probe first.

---

## PART 20 — RAG Quality: How Do We Know It Actually Works?

**Current state: there is no evaluation system.** `evaluation/README.md` says exactly that — *"Reserved for future retrieval and answer-quality evaluation datasets, metrics, and reports. No evaluation pipeline is implemented."* Nothing in the codebase measures retrieval quality, answer correctness, or latency. Every claim about "it works" today can only come from manually trying it in the browser.

**What's missing, concretely:**
- No labeled dataset of (question → expected relevant page/chunk → expected answer).
- No script that runs a batch of questions through the real pipeline and records results.
- No metric computation at all — not recall, not precision, not latency, not faithfulness.

**A practical, minimal evaluation approach for DocuChat:**

1. **Build a small ground-truth set.** Take the one real PDF already in this project (or a couple more), and manually write **20-50 questions**, each with:
   - The expected page number(s) that should be retrieved.
   - A short expected answer (or "not answerable from this document," for negative test cases).
2. **Run each question through the real `retrieve()` function** (this can be a standalone script in the currently-empty `scripts/` folder) and record which chunks/pages actually came back.
3. **Compute simple, honest metrics:**
   - **Retrieval recall** — of the expected page(s), what fraction appeared anywhere in the top-8 retrieved chunks?
   - **Retrieval precision** — of the top-8 retrieved chunks, what fraction were actually from an expected page?
   - **Answer correctness** — for each question, does the final generated answer match the expected answer (can start as a manual yes/no/partial judgment; later could be automated with an LLM-as-judge call)?
   - **Faithfulness** — does the generated answer only state things actually present in its cited excerpts (currently unchecked at runtime, per Part 18 #6 — this eval would be the first place to actually measure it)?
   - **Context relevance** — for the negative test cases ("not answerable"), does the system correctly say so, or does it hallucinate a plausible-sounding but ungrounded answer?
   - **Latency** — wall-clock time for ingestion (per page/chunk) and for a full question→answer round trip, since nothing currently measures this and the synchronous pipeline (Part 16) makes it worth watching.
4. **Store results as a simple committed table** (a CSV or a markdown table in `evaluation/`) — question, expected page, retrieved pages, expected answer, actual answer, pass/fail per metric.

**Why this matters beyond looking rigorous:** right now, "hybrid retrieval works well" and "citations are reliable" are claims that could only be verified by the developer's own casual testing. A committed evaluation report — even a small, imperfect 25-question one — is what turns those claims into something a skeptical reader (or interviewer) can actually trust, and it's exactly the kind of artifact that separates a demo project from an engineered one.

---

## PART 21 — Complete Request Lifecycle

### A. Document ingestion

```mermaid
flowchart TD
    A[User selects a PDF in the browser] --> B["POST /api/v1/documents (multipart upload)"]
    B --> C["documents.py: validate content-type/extension"]
    C --> D["Save raw file to data/uploads/{uuid}.pdf"]
    D --> E["Create Document row, status = processing"]
    E --> F["services/documents.py: process_document()"]
    F --> G["PyMuPDF (fitz): extract plain text per page"]
    G --> H["chunk_text(): ~900 chars, 150 overlap, sentence-boundary snap"]
    H --> I["embed_in_batches(): 16 chunks per call"]
    I --> J["Ollama /api/embed — qwen3-embedding:0.6b"]
    J --> K["ChromaDB collection.add(): vectors + text + metadata"]
    J --> L["SQLite: bulk insert Chunk rows"]
    K --> M["Document.status = ready, page_count set"]
    L --> M
    M --> N["HTTP 201 response to browser"]
```

### B. User question → answer

```mermaid
flowchart TD
    A[User types a question, hits send] --> B["POST /api/v1/chat"]
    B --> C{conversation_id given?}
    C -->|No| D[Create new Conversation row]
    C -->|Yes| E[Load existing Conversation]
    D --> F["retrieval.py: retrieve()"]
    E --> F
    F --> G["Load candidate chunks from SQLite (optionally filtered by document_ids)"]
    F --> H["Embed question with instruct prefix — Ollama qwen3-embedding:0.6b"]
    H --> I["ChromaDB cosine search, top 24"]
    F --> J["lexical_score(): keyword + phrase scoring over same candidates"]
    I --> K["Fuse rankings (RRF-style formula)"]
    J --> K
    K --> L["Take top 8 chunks as sources"]
    L --> M{Any sources found?}
    M -->|No| N["HTTP 400: Upload a text-based PDF before asking"]
    M -->|Yes| O["Build SOURCE n | file | page context block"]
    O --> P["Construct strict JSON-only prompt + system message"]
    P --> Q["Ollama /api/chat — qwen3:8b (non-streaming)"]
    Q --> R["grounded_response(): strip think-tags, parse/validate JSON"]
    R --> S{"Valid answer + >=1 valid source_id?"}
    S -->|No| T["Fixed refusal message, empty citations"]
    S -->|Yes| U["Build citation objects: index, filename, page, 280-char excerpt"]
    T --> V["Save user + assistant Message rows"]
    U --> V
    V --> W["JSON response to browser"]
    W --> X["React renders chat bubble + expandable citations"]
```

---

## PART 22 — "If I Have to Explain This in an Interview"

### 30-second version
"DocuChat is a local RAG app — you upload a PDF, it chunks and embeds the text with a local model, stores it in a vector database, and when you ask a question it retrieves the most relevant passages using both semantic and keyword search, then has a local LLM answer strictly from those passages with page citations. Everything runs on-device through Ollama — no cloud APIs."

### 2-minute version
"I built DocuChat to solve a specific annoyance: when you paste a long PDF into a chatbot, you can't tell where an answer actually came from, and it either doesn't fit or the model skims it and gets vague. So I built a small RAG pipeline instead. On upload, a FastAPI backend uses PyMuPDF to pull text out of the PDF, splits it into roughly 900-character overlapping chunks, and embeds each chunk with a local Qwen embedding model running through Ollama — so nothing leaves my machine. Those embeddings go into ChromaDB.

When you ask a question, I don't just do a plain vector search — I combine that with a keyword/phrase-matching pass I wrote by hand, and fuse the two rankings with a reciprocal-rank-fusion-style formula, so exact terms and semantic meaning both count. I take the top 8 chunks, build a strict prompt that says 'only answer from these passages, and tell me which ones you used, as JSON,' and send that to a local Qwen 8B model. Then I validate the model's own citations before showing anything to the user — if it claims a source that doesn't exist, or doesn't cite anything, I fall back to an honest 'couldn't find a supported answer' instead of showing something ungrounded.

The trickiest part was actually getting reliable citations out of a local model — it emits its own reasoning in `<think>` tags I have to strip, and its JSON isn't always perfectly formed, so I had to add a regex fallback path.

The main limitation right now is that everything is synchronous — a big PDF blocks the upload request while it's embedded — and there's no real evaluation yet; I've tested it by hand, but I don't have retrieval-recall or answer-correctness numbers. That's actually the next thing I want to add: a small labeled question set so I can measure retrieval quality instead of just eyeballing it."

### 5-minute technical outline
1. **Problem framing** — why RAG instead of long-context stuffing (Part 8).
2. **Ingestion pipeline** — extraction (PyMuPDF), chunking strategy and parameters, batched embedding, dual storage in Chroma + SQLite and why (Part 6, Part 10).
3. **Retrieval** — vector search, hand-written lexical scorer, RRF-style fusion, and why hybrid beats either alone (Part 7, Part 2).
4. **Generation and grounding** — the strict prompt contract, JSON parsing with fallback, citation validation, the refusal path (Part 7 steps 15-20).
5. **Architecture choices** — fully local via Ollama, SQLite by default with optional Postgres, no background jobs yet (Part 11, Part 12).
6. **One real challenge** — getting trustworthy structured output (citations) from a local "thinking" model that emits reasoning tags and imperfect JSON.
7. **One real limitation** — no evaluation harness yet; retrieval and faithfulness are unmeasured (Part 20).
8. **Next step** — a reranker and a small labeled evaluation set, because those are the two changes most likely to move actual answer quality (Part 19, Level 3/5).

---

## PART 23 — Interview Questions I Should Expect

### Beginner

**Q: What is RAG, and why did you use it instead of just asking an LLM directly?**
- *Testing:* whether you understand the core motivation, not just the acronym.
- *Simple answer:* RAG retrieves relevant text from your own documents first, then gives that text to the LLM as context, so the answer is grounded in real content instead of the model's memory. I used it because a plain LLM can't see my PDF's content at all unless I show it, and citing sources requires knowing exactly which passage was used.
- *Deeper answer:* it also solves the context-length and cost problem — my test document produced 1,407 chunks; sending the whole thing to an 8B local model on every question would be far slower and could exceed reasonable context limits, whereas retrieving only the top 8 relevant chunks keeps each request small and fast.

**Q: What's an embedding, and why do you need one?**
- *Testing:* basic conceptual grounding.
- *Simple answer:* it's a list of numbers representing the meaning of a piece of text, produced by a model (`qwen3-embedding:0.6b` here) — two pieces of text with similar meaning get similar number-lists, measured by cosine similarity.
- *Deeper answer:* I use an "instruct" prefix before embedding questions (but not chunks) because Qwen3's embedding model is trained for asymmetric retrieval — questions and passages are different kinds of text, and the instruct format tells the model which role this particular text is playing.

**Q: Why a vector database instead of just SQL?**
- *Testing:* whether you understand what vector databases are actually for.
- *Simple answer:* SQL is great at exact matches and simple filters, but terrible at "find the text most similar in meaning to this other text" — that requires comparing high-dimensional vectors efficiently, which is what ChromaDB's indexing is built for.
- *Deeper answer:* I actually use both — SQLite for structured lookups (documents, chunk text by ID, conversation history) and ChromaDB purely for the similarity search step, because each is good at a different kind of query.

### Intermediate

**Q: Why 900 characters with 150 overlap for chunking?**
- *Testing:* whether you can justify design choices or you'll admit when something wasn't rigorously tuned.
- *Simple answer:* honestly, that was a reasonable starting default, not a value I tuned against real evaluation data — I'd want to build a small labeled evaluation set (Part 20) and try 2-3 alternatives to actually justify it.
- *Deeper answer:* the overlap exists so a sentence or fact that falls right at a chunk boundary still appears intact in the next chunk, and the code also tries to snap the cut point to the nearest sentence-ending period rather than a hard character cutoff, to avoid slicing mid-sentence when possible.

**Q: Why this embedding model, and how did you evaluate it against alternatives?**
- *Simple answer:* `qwen3-embedding:0.6b` is small and fast enough to embed a large number of chunks locally without becoming the bottleneck, and it's designed to work well with the instruct-prefix retrieval pattern I'm using.
- *Deeper answer:* I haven't formally benchmarked it against alternatives like `nomic-embed-text` or a larger Qwen embedding variant — that's a gap I'd want to close with the same evaluation harness (Part 20), since embedding quality directly caps retrieval quality.

**Q: How does retrieval actually work end to end?**
- *Simple answer:* I embed the question, do a cosine similarity search in ChromaDB for the top 24 candidates, separately score the same candidates with a hand-written keyword/phrase matcher, and merge both rankings with a reciprocal-rank-fusion-style formula before taking the top 8.
- *Deeper answer:* RRF works by scoring each item as `1/(k + rank)` in each ranking and summing across rankings — it's rank-based rather than raw-score-based, which sidesteps the problem that vector similarity scores and keyword scores live on completely different numeric scales and can't be directly compared or averaged.

**Q: How do you reduce hallucination?**
- *Simple answer:* the prompt explicitly instructs the model to answer only from the supplied passages and to say so if it can't, and I require the response to name which passages (source_ids) it used — if it doesn't provide valid ones, I discard its answer and show a fixed "couldn't verify" message instead.
- *Deeper answer:* what I *don't* currently do is verify that the answer's content actually matches what the cited passage says — I only verify the citation *number* is valid, not that it's used honestly. That's a real gap (Part 8, Part 19 Level 4) I'd close with a faithfulness-check step.

### Advanced

**Q: How would you scale this to more concurrent users?**
- *Simple answer:* right now ingestion is synchronous inside the HTTP request and a new ChromaDB client opens per request — I'd move ingestion to a background task/queue, keep one long-lived Chroma client, and consider Postgres (already supported via `DATABASE_URL`) instead of SQLite once there's real concurrent write load.
- *Deeper answer:* I'd also need to introduce per-user document ownership, since right now there's no isolation at all — every document is globally visible — which is fine for one local user but breaks immediately with more than one.

**Q: How would you evaluate retrieval quality?**
- *Simple answer:* build a small labeled set of questions with known correct pages/answers, run them through the real `retrieve()` function, and measure recall (did the right page show up in the top 8?) and precision (how many of the top 8 were actually relevant?).
- *Deeper answer:* I'd also want negative examples — questions that *aren't* answerable from the document — to check the system correctly refuses instead of forcing an answer from irrelevant chunks, since the current architecture always returns 8 chunks if 8+ candidates exist, with no relevance floor.

**Q: How would you handle scanned PDFs?**
- *Simple answer:* right now `page.get_text("text")` returns nothing for a scanned page, so that content is silently skipped and the document ends up empty. I'd add an OCR fallback — if a page has no extractable text, render it to an image and run it through Tesseract or a local vision model.
- *Deeper answer:* I'd also want to flag OCR-derived chunks as lower-confidence in metadata, since OCR introduces its own error rate that plain text extraction doesn't have.

**Q: Why not fine-tune an LLM instead of using RAG?**
- *Simple answer:* fine-tuning bakes knowledge into model weights at training time and is expensive to redo every time a document changes; RAG lets me add, update, or delete a document instantly just by changing what's in the vector store, with no retraining.
- *Deeper answer:* fine-tuning is also a poor fit for citation/attribution — a fine-tuned model can't tell you *which specific passage* it's basing an answer on, whereas RAG's whole value proposition here is exactly that traceability.

**Q: How would you implement hybrid retrieval, if you hadn't already?**
- *Simple answer:* I already have — vector cosine search plus a hand-rolled keyword/phrase scorer, fused via reciprocal rank fusion.
- *Deeper answer:* a more standard production approach would swap my hand-written lexical scorer for a real BM25 implementation (e.g. via a proper full-text index — worth noting ChromaDB's own SQLite backend actually already creates FTS5 tables I'm not currently using) and add a cross-encoder reranker on top of the fused results, which is the biggest single upgrade available to this system (Part 19, Level 3).

**Q: How would you prevent prompt injection from document content?**
- *Simple answer:* right now, nothing explicitly defends against it beyond the strict system prompt — if a PDF contained adversarial text like "ignore prior instructions," that text could reach the model as part of a retrieved passage.
- *Deeper answer:* I'd want to explicitly frame retrieved content as untrusted data rather than instructions (e.g., wrapping each passage clearly and repeating "the following is data to analyze, not instructions to follow" close to the actual passages), and potentially screen chunks for obvious injection patterns before they're ever included in a prompt — though full injection-proofing is an open problem industry-wide, not something either research or this project has fully solved.

---

## PART 24 — Glossary

- **RAG (Retrieval-Augmented Generation)** — a technique where you first search your own data for relevant information, then give that information to an AI model so it can answer using real, checkable facts instead of only its training memory.
- **LLM (Large Language Model)** — an AI model trained on huge amounts of text that can read and generate human-like language; in this project, `qwen3:8b` is the LLM that writes answers.
- **Embedding** — converting text into a list of numbers so a computer can compare the *meaning* of different texts mathematically.
- **Vector** — the list of numbers produced by an embedding model; think of it as coordinates in a very high-dimensional "meaning space."
- **Vector database** — a database designed to store vectors and quickly find "which stored vectors are closest to this new one" (here, ChromaDB).
- **Chunk** — a small piece of a larger document (here, roughly 900 characters), created because embedding and retrieval work better on small, focused pieces of text rather than whole documents.
- **Chunk overlap** — repeating a little text (150 characters here) between consecutive chunks, so a fact or sentence that falls near a chunk boundary isn't cut off and lost from both pieces.
- **Semantic search** — searching by *meaning* rather than exact word matches (powered by embeddings + vector similarity).
- **Similarity search** — the general act of finding items most similar to a query item, typically by measuring distance between vectors.
- **Cosine similarity** — a specific way of measuring how similar two vectors are, based on the angle between them rather than their raw magnitude; this project's Chroma collection is explicitly configured to use it.
- **Retriever** — the part of a RAG system responsible for finding relevant chunks for a given question (here, `services/retrieval.py::retrieve()`).
- **Context** — the block of retrieved text handed to the LLM so it has something concrete to answer from (here, the `SOURCE n | file | page` blocks).
- **Prompt** — the actual text instructions and content sent to an LLM to get a response.
- **Inference** — the act of running a trained model to produce an output (as opposed to training it); every call to Ollama here is an inference call.
- **Token** — the small chunks (often sub-word pieces) that LLMs actually read and generate text in; not directly manipulated in this codebase, but relevant to model context limits and speed.
- **Temperature** — a setting controlling how random/creative vs. deterministic an LLM's output is; **not set anywhere in this project**, so Ollama's own default applies.
- **Metadata** — extra structured information attached to a piece of data; here, each chunk's metadata includes its document ID, filename, and page number.
- **API (Application Programming Interface)** — a defined way for one piece of software to talk to another; here, the backend exposes an API the frontend calls, and Ollama exposes an API the backend calls.
- **Endpoint** — one specific URL+method combination an API exposes, like `POST /api/v1/chat`.
- **Backend** — the server-side program handling data, logic, and storage (here, the FastAPI app).
- **Frontend** — the part of the app the user actually sees and clicks on in a browser (here, the Next.js/React app).
- **Middleware** — code that runs on every request/response passing through a web server before/after the main handler; here, FastAPI's CORS middleware.
- **OCR (Optical Character Recognition)** — technology that reads text out of images (e.g., a scanned page); **not implemented** in this project.
- **Multimodal** — an AI model or system that understands more than one type of input (e.g., text *and* images); this project is text-only.
- **Reranker** — a (typically more accurate, more expensive) model used to re-score an initial list of retrieved candidates to produce a better final ranking; **not present** in this project — a simpler heuristic fusion formula stands in for it.
- **Hybrid search** — combining two different search strategies (here, semantic vector search and keyword/lexical search) to get better results than either alone.
- **Hallucination** — when an AI model states something confidently that isn't actually true or isn't actually supported by its given evidence.
- **Grounding** — keeping an AI model's output tied to real, verifiable source material, rather than letting it answer purely from memory; the entire point of `grounded_response()` in this project.

---

## PART 25 — Final Project Map

```text
DOCUCHAT
│
├── Frontend
│   └── One Next.js/React page (page.tsx) + one API client module (lib/api.ts) — no component splitting yet
│
├── Backend
│   └── FastAPI app with 4 route groups: health, system, documents, chat
│
├── Ingestion
│   └── PyMuPDF text extraction → whitespace cleanup → 900/150-char chunking → batched Ollama embedding → dual write to ChromaDB + SQLite (fully synchronous, no background jobs)
│
├── Retrieval
│   └── Vector cosine search (top 24) + hand-written lexical/phrase scoring, fused via RRF-style formula, truncated to top 8 — no reranker model, no relevance floor
│
├── AI Models
│   ├── Embedding: qwen3-embedding:0.6b (local, via Ollama)
│   └── Generation: qwen3:8b (local, via Ollama, non-streaming)
│
├── Storage
│   ├── SQLite (default) or PostgreSQL (optional): documents, chunks, conversations, messages
│   └── ChromaDB (persistent, embedded): chunk vectors + metadata
│
└── Configuration
    └── One repo-root .env, read via pydantic-settings — no secrets, no per-user config, everything local
```

**When a PDF enters →** it's saved under a random UUID filename, read page-by-page with PyMuPDF, sliced into ~900-character overlapping chunks, embedded in batches of 16 through a local Ollama model, and written into both ChromaDB (for similarity search) and SQLite (for structured lookups) — all inside the single upload HTTP request.

**When a question enters →** it's embedded (with an instruct prefix), matched against stored chunks by both cosine similarity and a hand-written keyword scorer, the two rankings are fused, and the top 8 passages become the only evidence a local LLM is allowed to answer from.

**When the LLM answers →** its raw output is stripped of internal `<think>` reasoning, parsed as JSON, and every claimed citation is checked against the real retrieved sources before anything is shown to the user — an unverifiable or uncited answer is replaced with an honest refusal message rather than shown as-is.

**Main strength →** a genuinely working, fully local hybrid-retrieval RAG pipeline with real citation validation — not a demo stub, and unusually well-matched by its own README.

**Main limitation →** there is no evaluation of any kind (no measured recall, precision, or answer-correctness numbers exist anywhere in the repo), and ingestion is fully synchronous with no background processing, so both retrieval quality and large-document handling are currently unverified and unbounded, respectively.

**Most useful next improvement →** build the small evaluation harness described in Part 20. It's the one change that would let every other design decision in this document — the chunk size, the embedding model, the fusion formula, the citation logic — be judged by evidence instead of by reading the code and guessing.
