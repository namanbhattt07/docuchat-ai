import asyncio
import json

import fitz
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Chunk, Document, PageText
from app.services import documents as documents_module
from app.services import keyword_index
from app.services.documents import process_document


class FakeCollection:
    def __init__(self) -> None:
        self.added: list[dict] = []

    def add(self, ids, documents, embeddings, metadatas):
        self.added.append({"ids": ids, "documents": documents, "metadatas": metadatas})


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _build_pdf(path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter One", fontsize=24)
    page.insert_text((72, 110), "Machine learning is a subfield of artificial intelligence.", fontsize=11)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Chapter Two", fontsize=24)
    page2.insert_text((72, 110), "Deep learning uses layered neural networks.", fontsize=11)
    doc.save(path)
    doc.close()


def test_process_document_populates_toc_and_page_text(db_session, tmp_path, monkeypatch) -> None:
    async def fake_embed_in_batches(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", fake_embed_in_batches)

    pdf_path = tmp_path / "sample.pdf"
    _build_pdf(pdf_path)
    document = Document(id="doc-1", filename="sample.pdf", status="processing")
    db_session.add(document)
    db_session.commit()

    asyncio.run(process_document(db_session, document, pdf_path, FakeCollection()))

    assert document.status == "ready"
    assert document.page_count == 2
    toc = json.loads(document.toc_json)
    assert {item["title"] for item in toc} == {"Chapter One", "Chapter Two"}
    assert all(item["source"] == "heuristic" for item in toc)

    pages = db_session.scalars(select(PageText).where(PageText.document_id == "doc-1")).all()
    assert {page.page_number for page in pages} == {1, 2}
    page_one = next(page for page in pages if page.page_number == 1)
    assert "machine learning" in page_one.content.lower()

    chunks = db_session.scalars(select(Chunk).where(Chunk.document_id == "doc-1")).all()
    assert len(chunks) > 0
    assert all(chunk.bbox is not None for chunk in chunks)

    # Group 3: the keyword/FTS5 layer comes online in the same ingestion pass.
    hits = keyword_index.search(db_session, ["machine", "learning"], ["doc-1"], limit=10)
    assert hits

    # Suggestion generation is best-effort but must still run and cache.
    assert document.suggested_questions_json is not None


def test_process_document_generates_and_caches_suggested_questions(db_session, tmp_path, monkeypatch) -> None:
    async def fake_embed_in_batches(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", fake_embed_in_batches)

    captured = {}

    async def fake_generate_suggested_questions(page_texts, count):
        captured["page_texts"] = page_texts
        captured["count"] = count
        return ["What is machine learning?", "What is deep learning?"]

    monkeypatch.setattr(documents_module, "generate_suggested_questions", fake_generate_suggested_questions)

    pdf_path = tmp_path / "sample.pdf"
    _build_pdf(pdf_path)
    document = Document(id="doc-1", filename="sample.pdf", status="processing")
    db_session.add(document)
    db_session.commit()

    asyncio.run(process_document(db_session, document, pdf_path, FakeCollection()))

    assert json.loads(document.suggested_questions_json) == ["What is machine learning?", "What is deep learning?"]
    assert len(captured["page_texts"]) == 2


def test_process_document_caches_empty_suggestions_when_generation_fails(db_session, tmp_path, monkeypatch) -> None:
    async def fake_embed_in_batches(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", fake_embed_in_batches)

    async def failing_generate_suggested_questions(page_texts, count):
        raise RuntimeError("ollama is down")

    monkeypatch.setattr(documents_module, "generate_suggested_questions", failing_generate_suggested_questions)

    pdf_path = tmp_path / "sample.pdf"
    _build_pdf(pdf_path)
    document = Document(id="doc-1", filename="sample.pdf", status="processing")
    db_session.add(document)
    db_session.commit()

    asyncio.run(process_document(db_session, document, pdf_path, FakeCollection()))

    # A suggestion-generation failure must never fail ingestion or leave the
    # document un-ready.
    assert document.status == "ready"
    assert json.loads(document.suggested_questions_json) == []


def test_process_document_marks_empty_status_for_no_extractable_text(db_session, tmp_path, monkeypatch) -> None:
    async def fake_embed_in_batches(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", fake_embed_in_batches)

    doc = fitz.open()
    doc.new_page()
    pdf_path = tmp_path / "blank.pdf"
    doc.save(pdf_path)
    doc.close()

    document = Document(id="doc-2", filename="blank.pdf", status="processing")
    db_session.add(document)
    db_session.commit()

    asyncio.run(process_document(db_session, document, pdf_path, FakeCollection()))

    assert document.status == "empty"
    assert json.loads(document.toc_json) == []
    pages = db_session.scalars(select(PageText).where(PageText.document_id == "doc-2")).all()
    assert len(pages) == 1
    assert pages[0].content == ""
    assert json.loads(document.suggested_questions_json) == []
