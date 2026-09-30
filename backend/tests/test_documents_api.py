import json
from dataclasses import dataclass
from pathlib import Path

import fitz
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.documents as documents_api
from app.db import Base
from app.models import Chunk, Document, PageText


@dataclass
class FakeSettings:
    upload_directory: str
    suggested_questions_count: int = 5


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter One", fontsize=24)
    page.insert_text((72, 110), "Machine learning is a subfield of artificial intelligence.", fontsize=11)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Chapter Two", fontsize=24)
    page2.insert_text((72, 110), "Deep learning uses neural networks with many layers.", fontsize=11)
    doc.save(path)
    doc.close()


def _seed_document(session, document_id: str, upload_dir: Path) -> Document:
    _make_pdf(upload_dir / f"{document_id}.pdf")
    document = Document(id=document_id, filename="sample.pdf", status="ready")
    session.add(document)
    session.commit()
    return document


def test_get_document_toc_backfills_and_returns_heuristic_entries(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    document = _seed_document(db_session, "doc-1", tmp_path)
    assert document.toc_json is None  # not yet indexed

    response = documents_api.get_document_toc("doc-1", db=db_session)

    assert response["document_id"] == "doc-1"
    titles = [item["title"] for item in response["items"]]
    assert "Chapter One" in titles
    assert "Chapter Two" in titles
    assert document.toc_json is not None  # cached for next time


def test_get_document_toc_uses_cached_value_without_reparsing(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    document = _seed_document(db_session, "doc-1", tmp_path)
    document.toc_json = json.dumps([{"id": "native-0", "title": "Cached Entry", "page": 1, "level": 1, "source": "native"}])
    db_session.commit()

    # Delete the underlying PDF -- if the endpoint tried to re-parse it would
    # crash, so a successful cached response proves it didn't.
    (tmp_path / "doc-1.pdf").unlink()

    response = documents_api.get_document_toc("doc-1", db=db_session)

    assert response["items"][0]["title"] == "Cached Entry"


def test_get_document_toc_404s_for_unknown_document(db_session, tmp_path, monkeypatch) -> None:
    from fastapi import HTTPException

    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))

    with pytest.raises(HTTPException) as exc_info:
        documents_api.get_document_toc("missing-doc", db=db_session)

    assert exc_info.value.status_code == 404


def test_search_document_returns_case_insensitive_phrase_matches(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    _seed_document(db_session, "doc-1", tmp_path)

    response = documents_api.search_document("doc-1", q="MACHINE LEARNING", db=db_session)

    assert response["total"] == 1
    assert response["matches"][0]["page"] == 1
    assert response["matches"][0]["index"] == 1


def test_search_document_finds_matches_across_pages(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    _seed_document(db_session, "doc-1", tmp_path)

    response = documents_api.search_document("doc-1", q="learning", db=db_session)

    assert response["total"] == 2
    assert {match["page"] for match in response["matches"]} == {1, 2}


def test_search_document_returns_no_matches_for_absent_phrase(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    _seed_document(db_session, "doc-1", tmp_path)

    response = documents_api.search_document("doc-1", q="quantum entanglement", db=db_session)

    assert response["total"] == 0
    assert response["matches"] == []


def test_search_document_does_not_reparse_pdf_on_second_call(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    _seed_document(db_session, "doc-1", tmp_path)

    documents_api.search_document("doc-1", q="learning", db=db_session)
    (tmp_path / "doc-1.pdf").unlink()  # if a second search re-parsed, this would raise 404

    response = documents_api.search_document("doc-1", q="learning", db=db_session)

    assert response["total"] == 2


def test_delete_document_removes_page_text_rows(db_session, tmp_path, monkeypatch) -> None:
    class FakeCollection:
        def delete(self, where):
            pass

    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    monkeypatch.setattr(documents_api, "get_collection", lambda: FakeCollection())
    _seed_document(db_session, "doc-1", tmp_path)
    documents_api.search_document("doc-1", q="learning", db=db_session)  # populates PageText rows

    assert db_session.query(PageText).filter(PageText.document_id == "doc-1").count() > 0

    documents_api.delete_document("doc-1", db=db_session)

    assert db_session.query(PageText).filter(PageText.document_id == "doc-1").count() == 0


def test_delete_document_removes_keyword_index_rows(db_session, tmp_path, monkeypatch) -> None:
    class FakeCollection:
        def delete(self, where):
            pass

    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    monkeypatch.setattr(documents_api, "get_collection", lambda: FakeCollection())
    document = Document(id="doc-1", filename="sample.pdf", status="ready")
    db_session.add(document)
    db_session.add(Chunk(id="doc-1-1-0", document_id="doc-1", page_number=1, content="Machine learning overview."))
    db_session.commit()
    from app.services import keyword_index
    keyword_index.index_chunks(db_session, "doc-1", ["doc-1-1-0"], ["Machine learning overview."], [1])
    db_session.commit()
    assert keyword_index.search(db_session, ["machine"], ["doc-1"], limit=10) != []

    documents_api.delete_document("doc-1", db=db_session)

    assert keyword_index.search(db_session, ["machine"], ["doc-1"], limit=10) == []


def test_get_suggested_questions_returns_cached_value_without_reparsing(db_session, tmp_path, monkeypatch) -> None:
    import asyncio
    import json as json_module

    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))
    document = _seed_document(db_session, "doc-1", tmp_path)
    document.suggested_questions_json = json_module.dumps(["What is machine learning?", "What is deep learning?"])
    db_session.commit()
    (tmp_path / "doc-1.pdf").unlink()  # if this re-parsed, it would crash

    response = asyncio.run(documents_api.get_suggested_questions("doc-1", db=db_session))

    assert response["questions"] == ["What is machine learning?", "What is deep learning?"]


def test_get_suggested_questions_backfills_when_not_yet_generated(db_session, tmp_path, monkeypatch) -> None:
    import asyncio

    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))

    async def fake_generate_suggested_questions(page_texts, count):
        return ["What is machine learning?"]

    monkeypatch.setattr(documents_api, "generate_suggested_questions", fake_generate_suggested_questions)
    document = _seed_document(db_session, "doc-1", tmp_path)
    assert document.suggested_questions_json is None

    response = asyncio.run(documents_api.get_suggested_questions("doc-1", db=db_session))

    assert response["questions"] == ["What is machine learning?"]
    assert document.suggested_questions_json is not None


def test_get_suggested_questions_404s_for_unknown_document(db_session, tmp_path, monkeypatch) -> None:
    import asyncio

    from fastapi import HTTPException

    monkeypatch.setattr(documents_api, "get_settings", lambda: FakeSettings(upload_directory=str(tmp_path)))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(documents_api.get_suggested_questions("missing-doc", db=db_session))

    assert exc_info.value.status_code == 404
