import asyncio
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider, FakeOcrEngine
from pdfs import add_scanned_page, add_text_page, build_pdf
from fastapi import BackgroundTasks, UploadFile
from starlette.datastructures import Headers

import app.api.v1.chat as chat_module
import app.api.v1.documents as documents_api
import app.services.providers.ollama as ollama_provider_module
import app.services.retrieval as retrieval_module
from app.api.v1.chat import ChatRequest, ask
from app.core.config import get_settings
from app.models import Chunk, Document
from app.services.documents import embed_in_batches, process_document
from app.services.ocr import OcrLine
from app.services.pdf_provider import PdfDocumentProvider
from app.services.providers.ollama import OllamaEmbeddingProvider, OllamaLLMProvider, close_client
from app.services.providers import use_providers

# Group 8 -- performance behaviours that are deterministic enough to pin:
# what a generation request tells Ollama, that one connection is reused, that
# embedding parallelism is bounded, and that slow work (OCR) never freezes the
# event loop the rest of the app is served from. (Timings themselves --
# seconds per document -- are measured live, not asserted here.)


def _mock_ollama(monkeypatch, handler):
    seen: list[httpx.Request] = []

    def recording(request):
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        ollama_provider_module, "_client",
        httpx.AsyncClient(transport=httpx.MockTransport(recording), base_url="http://ollama.test"),
    )
    return seen


# ---------- generation requests ----------


def test_generation_requests_pin_the_context_window_and_keep_thinking_off(monkeypatch) -> None:
    """Ollama's default 4096-token window silently truncates a longer prompt
    from the front (dropping the instructions); the window must be explicit,
    and thinking must stay off (~5x slower when on, measured)."""
    seen = _mock_ollama(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": "ok"}}))
    settings = get_settings()

    asyncio.run(OllamaLLMProvider().generate([{"role": "user", "content": "hi"}]))

    body = json.loads(seen[0].content)
    assert body["options"] == {"num_ctx": settings.ollama_num_ctx}
    assert settings.ollama_num_ctx >= 8192  # the largest prompts the app builds are ~5.7k tokens
    assert body["think"] is False
    assert body["keep_alive"] == settings.ollama_keep_alive


def test_every_text_generation_asks_for_the_same_window(monkeypatch) -> None:
    """A different num_ctx between requests makes Ollama reload the model, so the
    value must not vary with the caller (rewrite / suggestions / answers alike)."""
    seen = _mock_ollama(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": "ok"}}))
    provider = OllamaLLMProvider()

    async def run():
        await provider.generate([{"role": "user", "content": "a"}])
        await provider.generate([{"role": "user", "content": "b"}], think=False)
        await OllamaLLMProvider().generate([{"role": "user", "content": "c" * 5000}])

    asyncio.run(run())

    assert {json.dumps(json.loads(request.content)["options"], sort_keys=True) for request in seen} == {json.dumps({"num_ctx": get_settings().ollama_num_ctx})}


def test_a_prompt_too_big_for_the_window_is_logged_as_a_truncation_risk(monkeypatch, caplog) -> None:
    _mock_ollama(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": "ok"}}))
    monkeypatch.setattr(get_settings(), "ollama_num_ctx", 4096)

    with caplog.at_level(logging.WARNING, logger="docuchat.ollama"):
        asyncio.run(OllamaLLMProvider().generate([{"role": "user", "content": "x" * 20000}]))
        asyncio.run(OllamaLLMProvider().generate([{"role": "user", "content": "short"}]))

    warnings = [record for record in caplog.records if "truncate" in record.getMessage()]
    assert len(warnings) == 1
    assert "OLLAMA_NUM_CTX" in warnings[0].getMessage()


# ---------- connection reuse ----------


class _KeepAliveServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        self.connections = 0
        self.requests = 0

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # persistent connections

            def do_POST(handler) -> None:  # noqa: N805
                handler.rfile.read(int(handler.headers.get("content-length", 0)))
                self.requests += 1
                payload = json.dumps({"embeddings": [[0.1, 0.2]]}).encode()
                handler.send_response(200)
                handler.send_header("content-type", "application/json")
                handler.send_header("content-length", str(len(payload)))
                handler.end_headers()
                handler.wfile.write(payload)

            def log_message(handler, *args) -> None:  # noqa: N805
                pass

        super().__init__(("127.0.0.1", 0), Handler)

    def get_request(self):
        self.connections += 1
        return super().get_request()


def test_many_model_calls_reuse_one_connection_even_from_fresh_provider_objects(monkeypatch) -> None:
    server = _KeepAliveServer()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(get_settings(), "ollama_base_url", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setattr(ollama_provider_module, "_client", None)

    async def run():
        try:
            for _ in range(15):
                # A new provider per call, as the vision path does -- the pool
                # is process-wide, not per provider object.
                await OllamaEmbeddingProvider().embed(["hello"])
        finally:
            await close_client()

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()

    assert server.requests == 15
    assert server.connections == 1


# ---------- embedding parallelism ----------


class _InFlightEmbedding:
    name = "tracking"

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def embed(self, texts):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return [[float(len(text))] for text in texts]


@pytest.mark.parametrize("concurrency", [1, 3])
def test_embedding_batches_run_in_parallel_up_to_the_limit_and_keep_their_order(concurrency) -> None:
    texts = ["x" * (index + 1) for index in range(40)]
    embedder = _InFlightEmbedding()

    with use_providers(embedding=embedder):
        vectors = asyncio.run(embed_in_batches(texts, batch_size=4, concurrency=concurrency))

    assert embedder.peak == concurrency  # parallel when allowed, never beyond the bound
    assert vectors == [[float(len(text))] for text in texts]  # one vector per text, in text order


# ---------- the event loop stays free ----------


def _slow_ocr(png: bytes):
    time.sleep(0.4)  # blocking, like a real CPU-bound OCR inference
    return [OcrLine("Station Delta measured a chlorine level of 0.8 mg per litre in March.", (60, 100, 1000, 140), 0.97)]


def test_ocr_runs_off_the_event_loop_so_the_app_keeps_answering(db_session, tmp_path) -> None:
    pdf = build_pdf(tmp_path / "scan.pdf", [lambda d: add_text_page(d), lambda d: add_scanned_page(d, "scanned words")])
    document = Document(id="doc-1", filename="scan.pdf", status="processing")
    db_session.add(document)
    db_session.commit()

    async def run() -> float:
        gaps: list[float] = []
        stop = asyncio.Event()

        async def ticker() -> None:  # stands in for every other request being served meanwhile
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        task = asyncio.create_task(ticker())
        provider = PdfDocumentProvider(ocr_engine=FakeOcrEngine(_slow_ocr))
        with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
            await process_document(db_session, document, pdf, _NoopCollection(), provider=provider)
        stop.set()
        await task
        return max(gaps)

    longest_stall = asyncio.run(run())

    assert document.status == "ready"
    assert longest_stall < 0.2, f"the event loop was blocked for {longest_stall:.2f}s while OCR ran"


class _NoopCollection:
    def add(self, ids, documents, embeddings, metadatas):
        pass


def test_upload_returns_immediately_and_leaves_ingestion_to_a_background_task(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "upload_directory", str(tmp_path / "uploads"))
    content = build_pdf(tmp_path / "a.pdf", [lambda d: add_text_page(d)]).read_bytes()
    upload = UploadFile(file=io.BytesIO(content), filename="a.pdf", headers=Headers({"content-type": "application/pdf"}))
    tasks = BackgroundTasks()

    result = asyncio.run(documents_api.upload_document(tasks, upload, db_session))

    assert result["status"] == "processing"
    assert [task.func.__name__ for task in tasks.tasks] == ["_run_ingestion"]  # queued, not awaited
    assert db_session.query(Chunk).count() == 0  # nothing has been indexed yet at response time


# ---------- vision / text separation ----------


class _NoCollection:
    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


def test_an_ordinary_question_never_touches_the_vision_model_or_its_capability_lookup(db_session, monkeypatch) -> None:
    """Asking about text must not pay for (or be broken by) the vision model:
    no /api/tags capability call, no vision provider, so no model swap."""
    monkeypatch.setattr(chat_module, "get_collection", lambda: _NoCollection())

    async def fake_embed(texts):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(retrieval_module, "embed", fake_embed)

    async def forbidden(*args, **kwargs):
        raise AssertionError("the vision stack must not be consulted for a text question")

    monkeypatch.setattr(chat_module, "vision_capability", forbidden)
    monkeypatch.setattr(chat_module, "get_vision_provider", forbidden)
    db_session.add(Document(id="doc-1", filename="protocols.pdf", status="ready", page_count=1))
    db_session.add(Chunk(id="doc-1-1-0", document_id="doc-1", page_number=1, content="ZigBee operates in the 2.4 GHz band.", start_offset=0, end_offset=36))
    db_session.commit()

    async def answer(messages, **kwargs):
        return json.dumps({"answer": "ZigBee uses 2.4 GHz. [1]", "source_ids": [1]})

    monkeypatch.setattr(chat_module, "generate", answer)
    result = asyncio.run(ask(ChatRequest(question="What frequency does ZigBee use?", document_ids=["doc-1"]), db=db_session))

    assert "2.4 GHz" in result["answer"]
