import asyncio
import json

import fitz
import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider
from pdfs import LOREM, add_text_page, build_pdf
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import sessionmaker

import app.api.v1.documents as documents_api
import app.services.documents as documents_module
from app.db import Base
from app.models import Chunk, Document, DocumentImage, PageText
from app.services.documents import INTERRUPTED_DETAIL, process_document, recover_interrupted_documents, suggestions_pending
from app.services.providers import ProviderUnavailable, use_providers

# Group 8 -- document lifecycle. What a person sees while a PDF goes through
# processing -> ready / failed / empty, and that the backend's stored state
# always agrees with what actually happened (no half-indexed failures, nothing
# stuck on "processing", nothing left behind by a delete).


class RecordingCollection:
    def __init__(self) -> None:
        self.added: list[list[str]] = []
        self.deleted: list[dict] = []

    def add(self, ids, documents, embeddings, metadatas):
        self.added.append(list(ids))

    def delete(self, where):
        self.deleted.append(where)

    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def sessions(tmp_path):
    """A file-backed database with a session factory, so a test can hold two
    independent sessions (the ingestion one and a "browser" one)."""
    engine = create_engine(f"sqlite:///{tmp_path}/lifecycle.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


def _text_pdf(tmp_path, name="doc.pdf"):
    return build_pdf(tmp_path / name, [lambda d: add_text_page(d, LOREM), lambda d: add_text_page(d, LOREM, heading="Methods")])


def _add_document(factory, document_id="doc-1", status="processing") -> None:
    with factory() as db:
        db.add(Document(id=document_id, filename=f"{document_id}.pdf", status=status))
        db.commit()


def _run_ingestion(factory, monkeypatch, document_id, path, collection=None):
    collection = collection or RecordingCollection()
    monkeypatch.setattr(documents_api, "SessionLocal", factory)
    monkeypatch.setattr(documents_api, "get_collection", lambda: collection)
    asyncio.run(documents_api._run_ingestion(document_id, path))
    return collection


def _rows(factory, document_id="doc-1"):
    with factory() as db:
        document = db.get(Document, document_id)
        counts = {
            model.__name__: len(db.scalars(select(model).where(model.document_id == document_id)).all())
            for model in (Chunk, PageText, DocumentImage)
        }
        return (document.status, document.status_detail) if document else None, counts


# ---------- ready means ready: not "ready once the suggestions are written" ----------


class _ObservingLLM:
    """A model that, while it is 'thinking' about suggested questions, looks at
    the database the way the browser's poll would."""

    name = "observing"

    def __init__(self, factory, document_id) -> None:
        self.factory, self.document_id = factory, document_id
        self.seen: dict = {}

    async def generate(self, messages, *, think=None):
        with self.factory() as reader:
            document = reader.get(Document, self.document_id)
            self.seen["status"] = document.status
            self.seen["chunks"] = len(reader.scalars(select(Chunk).where(Chunk.document_id == self.document_id)).all())
            self.seen["suggestions"] = document.suggested_questions_json
        self.seen["pending"] = suggestions_pending(self.document_id)
        return json.dumps(["What is machine learning?", "How does supervised learning work?"])

    async def status(self):
        return {}


def test_document_is_ready_and_searchable_while_suggested_questions_are_still_being_written(sessions, tmp_path) -> None:
    pdf = _text_pdf(tmp_path)
    _add_document(sessions)
    llm = _ObservingLLM(sessions, "doc-1")

    with sessions() as db, use_providers(embedding=FakeEmbeddingProvider(), llm=llm):
        asyncio.run(process_document(db, db.get(Document, "doc-1"), pdf, RecordingCollection()))

    # What another session (the polling UI) saw at the moment the model was busy:
    assert llm.seen["status"] == "ready"
    assert llm.seen["chunks"] > 0
    assert llm.seen["suggestions"] is None  # not written yet -- and that must not hold the status back
    assert llm.seen["pending"] is True
    # ...and once generation finished the questions are stored and nothing is "pending".
    with sessions() as db:
        assert json.loads(db.get(Document, "doc-1").suggested_questions_json) == ["What is machine learning?", "How does supervised learning work?"]
    assert suggestions_pending("doc-1") is False


def test_suggestions_endpoint_reports_pending_instead_of_starting_a_second_generation(sessions, tmp_path, monkeypatch) -> None:
    pdf = _text_pdf(tmp_path)
    _add_document(sessions)
    monkeypatch.setattr(documents_api, "_require_stored_pdf", lambda document_id: pdf)
    responses: dict = {}

    class _AskingLLM(_ObservingLLM):
        async def generate(self, messages, *, think=None):
            with sessions() as request_db:
                responses["during"] = await documents_api.get_suggested_questions("doc-1", request_db)
            return await super().generate(messages, think=think)

    with sessions() as db, use_providers(embedding=FakeEmbeddingProvider(), llm=_AskingLLM(sessions, "doc-1")):
        asyncio.run(process_document(db, db.get(Document, "doc-1"), pdf, RecordingCollection()))

    assert responses["during"] == {"document_id": "doc-1", "questions": [], "pending": True}
    with sessions() as db:
        after = asyncio.run(documents_api.get_suggested_questions("doc-1", db))
    assert after["questions"] == ["What is machine learning?", "How does supervised learning work?"]
    assert "pending" not in after


def test_a_failing_suggestion_call_still_leaves_a_ready_document(sessions, tmp_path) -> None:
    pdf = _text_pdf(tmp_path)
    _add_document(sessions)

    with sessions() as db, use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider(ProviderUnavailable("model gone"))):
        asyncio.run(process_document(db, db.get(Document, "doc-1"), pdf, RecordingCollection()))

    (status, detail), counts = _rows(sessions)
    assert status == "ready" and detail is None
    assert counts["Chunk"] > 0
    assert suggestions_pending("doc-1") is False


# ---------- failure: a reason the person can act on, and no half-indexed remains ----------


def _encrypted_pdf(tmp_path):
    doc = fitz.open()
    add_text_page(doc, LOREM)
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    return path


def _damaged_pdf(tmp_path):
    path = tmp_path / "damaged.pdf"
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nthis is not a real pdf\n")
    return path


@pytest.mark.parametrize("make_pdf, expected", [
    (_encrypted_pdf, "password-protected"),
    (_damaged_pdf, "no readable pages"),
])
def test_unreadable_pdfs_fail_with_the_reason_not_as_an_empty_document(sessions, tmp_path, monkeypatch, make_pdf, expected) -> None:
    pdf = make_pdf(tmp_path)
    _add_document(sessions)

    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        _run_ingestion(sessions, monkeypatch, "doc-1", pdf)

    (status, detail), counts = _rows(sessions)
    assert status == "failed"  # never "empty": a locked or broken file is not a blank one
    assert expected in detail
    assert counts == {"Chunk": 0, "PageText": 0, "DocumentImage": 0}


@pytest.mark.parametrize("failure, expected_detail", [
    (ProviderUnavailable("Could not reach Ollama's embedding service: refused"), "Start Ollama"),
    (RuntimeError("boom"), "could not be processed"),
])
def test_a_failure_after_pages_were_parsed_leaves_no_half_indexed_document(sessions, tmp_path, monkeypatch, failure, expected_detail) -> None:
    pdf = _text_pdf(tmp_path)
    _add_document(sessions)

    async def failing_embed(texts, **kwargs):
        raise failure

    monkeypatch.setattr(documents_module, "embed_in_batches", failing_embed)
    with use_providers(llm=FakeLLMProvider("[]")):
        collection = _run_ingestion(sessions, monkeypatch, "doc-1", pdf)

    (status, detail), counts = _rows(sessions)
    assert status == "failed"
    assert expected_detail in detail
    # The page text / figures / TOC were staged before the embedding call; the
    # failure must not commit them alongside "failed".
    assert counts == {"Chunk": 0, "PageText": 0, "DocumentImage": 0}
    assert collection.added == []


def test_a_document_deleted_while_processing_leaves_nothing_behind(sessions, tmp_path, monkeypatch) -> None:
    pdf = _text_pdf(tmp_path)
    _add_document(sessions)

    async def embed_then_get_deleted(texts, **kwargs):
        with sessions() as other:  # the user hits "Remove" while the PDF is being embedded
            other.execute(delete(Document).where(Document.id == "doc-1"))
            other.commit()
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", embed_then_get_deleted)
    with use_providers(llm=FakeLLMProvider("[]")):
        collection = _run_ingestion(sessions, monkeypatch, "doc-1", pdf)

    assert collection.added == []  # no orphan vectors for a document that no longer exists
    _, counts = _rows(sessions)
    assert counts == {"Chunk": 0, "PageText": 0, "DocumentImage": 0}
    with sessions() as db:
        assert db.get(Document, "doc-1") is None


def test_a_document_removed_while_its_suggestions_are_written_is_not_reported_as_a_failed_ingestion(sessions, tmp_path, monkeypatch, caplog) -> None:
    """Found in the browser: pressing Remove right after a document turned Ready
    (while its suggested questions were still being generated) made the
    bookkeeping read an attribute of the deleted row and log a false
    'Ingestion failed' traceback for a document that had been ingested fine."""
    pdf = _text_pdf(tmp_path)
    _add_document(sessions)

    class _RemovedWhileThinking:
        name = "removed-while-thinking"

        async def generate(self, messages, *, think=None):
            with sessions() as other:
                other.execute(delete(Document).where(Document.id == "doc-1"))
                other.commit()
            return json.dumps(["What is machine learning?"])

        async def status(self):
            return {}

    with caplog.at_level("ERROR"), use_providers(embedding=FakeEmbeddingProvider(), llm=_RemovedWhileThinking()):
        _run_ingestion(sessions, monkeypatch, "doc-1", pdf)

    assert "Ingestion failed" not in caplog.text
    assert suggestions_pending("doc-1") is False  # the in-flight marker is always released


# ---------- restart: nothing stays on "processing" forever ----------


def test_documents_left_processing_by_a_restart_are_failed_with_what_to_do(sessions) -> None:
    _add_document(sessions, "stuck", "processing")
    _add_document(sessions, "fine", "ready")
    _add_document(sessions, "blank", "empty")
    collection = RecordingCollection()

    with sessions() as db:
        recovered = recover_interrupted_documents(db, lambda: collection)

    assert recovered == 1
    assert _rows(sessions, "stuck")[0] == ("failed", INTERRUPTED_DETAIL)
    assert _rows(sessions, "fine")[0] == ("ready", None)
    assert _rows(sessions, "blank")[0] == ("empty", None)
    assert collection.deleted == [{"document_id": "stuck"}]  # any partial vectors are discarded


def test_startup_recovery_does_not_open_the_vector_store_when_nothing_is_stuck(sessions) -> None:
    _add_document(sessions, "fine", "ready")

    def must_not_open():
        raise AssertionError("the vector store should not be opened when there is nothing to recover")

    with sessions() as db:
        assert recover_interrupted_documents(db, must_not_open) == 0


# ---------- reading a document that is still being processed must not break its processing ----------


def test_opening_contents_or_search_mid_processing_no_longer_kills_the_ingestion(sessions, tmp_path, monkeypatch) -> None:
    """Regression: the TOC / search endpoints backfilled their cache when it was
    empty -- which it is for every document still mid-ingestion. That backfill
    raced the ingestion's own write to the same page_text rows, and the
    ingestion lost (UNIQUE constraint failed -> the document ended "failed")."""
    from fastapi import HTTPException

    pdf = _text_pdf(tmp_path)
    _add_document(sessions)
    monkeypatch.setattr(documents_api, "_require_stored_pdf", lambda document_id: pdf)
    poked: list[tuple[str, int]] = []

    async def embed_while_the_user_clicks_around(texts, **kwargs):
        with sessions() as request_db:
            for name, call in (
                ("toc", lambda: documents_api.get_document_toc("doc-1", request_db)),
                ("search", lambda: documents_api.search_document("doc-1", "learning", request_db)),
            ):
                with pytest.raises(HTTPException) as refused:
                    call()
                poked.append((name, refused.value.status_code))
                assert "still being processed" in refused.value.detail
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", embed_while_the_user_clicks_around)
    with use_providers(llm=FakeLLMProvider("[]")):
        _run_ingestion(sessions, monkeypatch, "doc-1", pdf)

    assert poked == [("toc", 409), ("search", 409)]
    (status, detail), counts = _rows(sessions)
    assert status == "ready" and detail is None
    assert counts["PageText"] == 2 and counts["Chunk"] > 0


def test_a_failed_document_has_no_contents_to_serve(sessions) -> None:
    from fastapi import HTTPException

    _add_document(sessions, "bad", "failed")
    with sessions() as db, pytest.raises(HTTPException) as refused:
        documents_api.get_document_toc("bad", db)

    assert refused.value.status_code == 409 and "could not be processed" in refused.value.detail
    with sessions() as db, pytest.raises(HTTPException) as refused:
        documents_api.search_document("bad", "anything", db)
    assert refused.value.status_code == 409
