import json
from pathlib import Path

import httpx
import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider
from fastapi.testclient import TestClient
from pdfs import LOREM, add_text_page, build_pdf
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.api.v1.chat as chat_module
import app.api.v1.documents as documents_api
from app.core.config import get_settings
from app.db import Base, get_db
from app.main import app
from app.models import Chunk, Document
from app.services.providers import ProviderUnavailable, use_providers

# Group 8 -- what the upload endpoint tells a person about a file it can't or
# won't take (unsupported / empty / duplicate), the happy path from upload to
# ready, and that a model outage reaches the browser as a readable error.


class _Collection:
    def add(self, ids, documents, embeddings, metadatas):
        pass

    def delete(self, where):
        pass

    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/api.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)

    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(get_settings(), "upload_directory", str(tmp_path / "uploads"))
    monkeypatch.setattr(documents_api, "SessionLocal", factory)
    monkeypatch.setattr(documents_api, "get_collection", lambda: _Collection())
    monkeypatch.setattr(chat_module, "get_collection", lambda: _Collection())
    test_client = TestClient(app)
    test_client.session_factory = factory
    test_client.upload_dir = tmp_path / "uploads"
    yield test_client
    app.dependency_overrides.pop(get_db, None)
    engine.dispose()


def _pdf_bytes(tmp_path: Path, text: str = LOREM) -> bytes:
    return build_pdf(tmp_path / "upload.pdf", [lambda d: add_text_page(d, text)]).read_bytes()


def _upload(client, name, content, content_type="application/pdf"):
    return client.post("/api/v1/documents", files={"file": (name, content, content_type)})


def _documents(client):
    return client.get("/api/v1/documents").json()


def _stored_files(client):
    return list(client.upload_dir.glob("*")) if client.upload_dir.exists() else []


# ---------- unsupported / empty files are rejected at the door, leaving nothing behind ----------


@pytest.mark.parametrize("name, content, content_type, status, message", [
    ("notes.txt", b"just some text", "text/plain", 415, "Only PDF files are supported"),
    ("renamed.pdf", b"this is plain text pretending to be a pdf", "application/pdf", 415, "isn't a PDF"),
    ("photo.pdf", b"\x89PNG\r\n\x1a\n" + b"0" * 200, "application/pdf", 415, "isn't a PDF"),
    ("empty.pdf", b"", "application/pdf", 400, "empty"),
])
def test_files_that_are_not_pdfs_are_refused_immediately_with_a_reason(client, name, content, content_type, status, message) -> None:
    response = _upload(client, name, content, content_type)

    assert response.status_code == status
    assert message in response.json()["detail"]
    assert _documents(client) == []  # no ghost row waiting in the library
    assert _stored_files(client) == []  # and no orphan file on disk


def test_a_real_pdf_passes_the_header_check_even_with_leading_junk(client, tmp_path) -> None:
    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        response = _upload(client, "junk-first.pdf", b"\n\n  junk before the header \n" + _pdf_bytes(tmp_path))

    assert response.status_code == 202


def test_a_file_over_the_size_limit_is_refused_with_the_limit_named(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "max_upload_size_mb", 0)
    response = _upload(client, "big.pdf", _pdf_bytes(tmp_path))

    assert response.status_code == 413
    assert "0 MB upload limit" in response.json()["detail"]
    assert _documents(client) == [] and _stored_files(client) == []


def test_uploading_the_same_pdf_twice_names_the_document_it_duplicates(client, tmp_path) -> None:
    content = _pdf_bytes(tmp_path)
    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        assert _upload(client, "report.pdf", content).status_code == 202
        second = _upload(client, "copy-of-report.pdf", content)

    assert second.status_code == 409
    assert 'already in your library as "report.pdf"' in second.json()["detail"]
    assert [d["filename"] for d in _documents(client)] == ["report.pdf"]
    assert len(_stored_files(client)) == 1


def test_a_failed_document_does_not_block_uploading_the_same_file_again(client, tmp_path) -> None:
    content = _pdf_bytes(tmp_path)
    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        first = _upload(client, "report.pdf", content).json()
    with client.session_factory() as db:
        document = db.get(Document, first["id"])
        document.status, document.status_detail = "failed", "This PDF could not be processed."
        db.commit()

    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        again = _upload(client, "report.pdf", content)

    assert again.status_code == 202


# ---------- happy path: upload -> processing -> ready ----------


def test_an_uploaded_pdf_ends_up_ready_with_its_pages_counted_and_its_chunks_stored(client, tmp_path) -> None:
    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider(json.dumps(["What is machine learning?"]))):
        response = _upload(client, "ml.pdf", _pdf_bytes(tmp_path))

    body = response.json()
    assert response.status_code == 202 and body["status"] == "processing"  # what the UI shows first
    (listed,) = _documents(client)  # the background task ran to completion by now
    assert listed["status"] == "ready" and listed["status_detail"] is None
    assert listed["page_count"] == 1
    assert listed["page_summary"]["text"] == 1
    with client.session_factory() as db:
        assert db.scalars(select(Chunk).where(Chunk.document_id == body["id"])).all()
    suggestions = client.get(f"/api/v1/documents/{body['id']}/suggested-questions").json()
    assert suggestions["questions"] == ["What is machine learning?"]


def test_a_blank_pdf_ends_up_empty_not_failed(client, tmp_path) -> None:
    import fitz

    doc = fitz.open()
    doc.new_page()
    doc.save(tmp_path / "blank.pdf")
    doc.close()
    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        _upload(client, "blank.pdf", (tmp_path / "blank.pdf").read_bytes())

    (listed,) = _documents(client)
    assert listed["status"] == "empty"
    assert "No readable text" in listed["status_detail"]


def test_a_password_protected_pdf_ends_up_failed_with_the_reason(client, tmp_path) -> None:
    import fitz

    doc = fitz.open()
    add_text_page(doc, LOREM)
    doc.save(tmp_path / "locked.pdf", encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        _upload(client, "locked.pdf", (tmp_path / "locked.pdf").read_bytes())

    (listed,) = _documents(client)
    assert listed["status"] == "failed"
    assert "password-protected" in listed["status_detail"]


# ---------- a model outage during retrieval is a readable error, not a network failure ----------


class _DownEmbedding:
    name = "down"

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def embed(self, texts):
        raise self.error


def _seed_readable_document(client) -> str:
    with client.session_factory() as db:
        db.add(Document(id="doc-1", filename="protocols.pdf", status="ready", page_count=1))
        db.add(Chunk(id="doc-1-1-0", document_id="doc-1", page_number=1, content="ZigBee operates in the 2.4 GHz band.", start_offset=0, end_offset=36))
        db.commit()
    return "doc-1"


def test_when_the_embedding_model_is_down_chat_returns_a_503_the_browser_can_read(client) -> None:
    document_id = _seed_readable_document(client)
    origin = get_settings().cors_origin_list[0]
    down = _DownEmbedding(ProviderUnavailable("Could not reach Ollama's embedding service: refused"))

    with use_providers(embedding=down, llm=FakeLLMProvider("unused")):
        response = client.post(
            "/api/v1/chat", json={"question": "What frequency does ZigBee use?", "document_ids": [document_id]}, headers={"Origin": origin},
        )

    assert response.status_code == 503
    assert "Start Ollama" in response.json()["detail"]
    # Without this header the browser hides the response and reports "Failed to fetch".
    assert response.headers["access-control-allow-origin"] == origin


def test_a_model_timeout_is_a_504_not_a_503(client) -> None:
    document_id = _seed_readable_document(client)
    error = ProviderUnavailable("timed out")
    error.__cause__ = httpx.ReadTimeout("slow")

    with use_providers(embedding=_DownEmbedding(error), llm=FakeLLMProvider("unused")):
        response = client.post("/api/v1/chat", json={"question": "What frequency does ZigBee use?", "document_ids": [document_id]})

    assert response.status_code == 504
    assert "too long" in response.json()["detail"]


# ---------- system status names what is missing ----------


def test_system_status_names_the_models_so_the_ui_can_say_which_one_is_missing(client, monkeypatch) -> None:
    import app.services.providers.ollama as ollama_module

    def only_the_embedding_model(request):
        return httpx.Response(200, json={"models": [{"name": get_settings().ollama_embedding_model}]})

    monkeypatch.setattr(ollama_module, "_client", httpx.AsyncClient(transport=httpx.MockTransport(only_the_embedding_model), base_url="http://ollama.test"))
    ollama = client.get("/api/v1/system/status").json()["ollama"]

    assert ollama["available"] is True
    assert ollama["generation_model_ready"] is False and ollama["embedding_model_ready"] is True
    assert ollama["generation_model"] == get_settings().ollama_generation_model
    assert ollama["embedding_model"] == get_settings().ollama_embedding_model


def test_system_status_when_ollama_is_not_running_still_answers(client, monkeypatch) -> None:
    import app.services.providers.ollama as ollama_module

    def refuse(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama_module, "_client", httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://ollama.test"))
    status = client.get("/api/v1/system/status")

    assert status.status_code == 200
    assert status.json()["ollama"]["available"] is False


# ---------- serving the stored PDF to the viewer (Group 1) ----------


def _stored_document(client, tmp_path, filename="Quarterly Report.pdf") -> tuple[str, bytes]:
    content = _pdf_bytes(tmp_path)
    client.upload_dir.mkdir(parents=True, exist_ok=True)
    with client.session_factory() as db:
        document = Document(filename=filename, status="ready", page_count=1)
        db.add(document)
        db.commit()
        document_id = document.id
    (client.upload_dir / f"{document_id}.pdf").write_bytes(content)
    return document_id, content


def test_the_stored_pdf_is_served_inline_with_its_exact_bytes(client, tmp_path) -> None:
    document_id, content = _stored_document(client, tmp_path)

    response = client.get(f"/api/v1/documents/{document_id}/file")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline")
    assert response.content == content


def test_range_requests_are_honoured_so_a_large_pdf_can_start_rendering_before_it_finishes_downloading(client, tmp_path) -> None:
    document_id, content = _stored_document(client, tmp_path)

    partial = client.get(f"/api/v1/documents/{document_id}/file", headers={"Range": "bytes=0-99"})

    assert partial.status_code == 206
    assert partial.content == content[:100]
    assert partial.headers["content-range"] == f"bytes 0-99/{len(content)}"
    assert client.get(f"/api/v1/documents/{document_id}/file").headers["accept-ranges"] == "bytes"


def test_the_viewer_on_another_origin_can_read_the_range_headers_pdfjs_needs(client, tmp_path) -> None:
    """The frontend (port 3000) and API (port 8000) are different origins, and a
    browser only lets a page read CORS-*exposed* response headers. Without
    Accept-Ranges / Content-Range exposed, pdf.js cannot tell the server supports
    ranges and silently downloads the whole file before drawing page one."""
    document_id, _ = _stored_document(client, tmp_path)
    origin = get_settings().cors_origin_list[0]

    response = client.get(f"/api/v1/documents/{document_id}/file", headers={"Origin": origin, "Range": "bytes=0-9"})

    exposed = {name.strip().lower() for name in response.headers["access-control-expose-headers"].split(",")}
    assert {"accept-ranges", "content-range", "content-length"} <= exposed


def test_download_forces_a_save_with_the_original_filename(client, tmp_path) -> None:
    document_id, _ = _stored_document(client, tmp_path, filename="Quarterly Report.pdf")

    response = client.get(f"/api/v1/documents/{document_id}/file", params={"download": 1})

    # Starlette uses the RFC 5987/6266 extended form, percent-encoding the name
    # (`filename*=utf-8''Quarterly%20Report.pdf`), which is what lets a non-ASCII
    # original name survive; browsers decode it back to "Quarterly Report.pdf".
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment")
    assert "filename*=utf-8''Quarterly%20Report.pdf" in disposition


def test_a_document_whose_file_is_missing_or_unknown_is_a_clear_404(client, tmp_path) -> None:
    document_id, _ = _stored_document(client, tmp_path)
    (client.upload_dir / f"{document_id}.pdf").unlink()

    missing_file = client.get(f"/api/v1/documents/{document_id}/file")
    unknown = client.get("/api/v1/documents/no-such-document/file")

    assert missing_file.status_code == 404 and "stored PDF" in missing_file.json()["detail"]
    assert unknown.status_code == 404 and "Document not found" in unknown.json()["detail"]
