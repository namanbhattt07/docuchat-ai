import asyncio
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.chat as chat_module
import app.services.retrieval as retrieval_module
from app.api.v1.chat import ChatRequest, ask
from app.db import Base
from app.models import Chunk, Document
from app.services import collections as collections_service

# Group 5: collection-scoped retrieval. Exercises the same /chat pipeline as
# test_chat.py, but focused on collection_id + partial document selection --
# see PARTIAL-COLLECTION SELECTION / RETRIEVAL SCOPE / SECURITY in the brief.


class _FakeCollection:
    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def seeded_db(db_session, monkeypatch):
    monkeypatch.setattr(chat_module, "get_collection", lambda: _FakeCollection())

    async def _fake_embed(texts):
        return [[0.0] * 4 for _ in texts]
    monkeypatch.setattr(retrieval_module, "embed", _fake_embed)

    async def _fake_generate(messages, **kwargs):
        return json.dumps({"answer": "ZigBee uses the 2.4 GHz band. [1]", "source_ids": [1]})
    monkeypatch.setattr(chat_module, "generate", _fake_generate)

    db_session.add_all([
        Document(id="unit-1", filename="Unit 1.pdf", status="ready"),
        Document(id="unit-2", filename="Unit 2.pdf", status="ready"),
        Document(id="outside", filename="Outside.pdf", status="ready"),
    ])
    db_session.add_all([
        Chunk(id="unit-1-59-0", document_id="unit-1", page_number=59, content="ZigBee operates in the 2.4 GHz band and uses mesh topology.", start_offset=0, end_offset=60),
        Chunk(id="unit-2-23-0", document_id="unit-2", page_number=23, content="Bluetooth Low Energy targets short-range, battery-powered devices.", start_offset=0, end_offset=60),
        Chunk(id="outside-1-0", document_id="outside", page_number=1, content="This document should never influence a scoped collection query.", start_offset=0, end_offset=60),
    ])
    db_session.commit()

    collection = collections_service.create_collection(db_session, "IoT Notes")
    collections_service.add_document(db_session, collection["id"], "unit-1")
    collections_service.add_document(db_session, collection["id"], "unit-2")
    return db_session, collection["id"]


def test_collection_retrieval_searches_across_member_documents(seeded_db) -> None:
    db, collection_id = seeded_db
    request = ChatRequest(question="What are the main differences between ZigBee and Bluetooth?", document_ids=["unit-1", "unit-2"], collection_id=collection_id)

    result = asyncio.run(ask(request, db=db))

    assert result["citations"]
    assert result["citations"][0]["document_id"] in {"unit-1", "unit-2"}


def test_unchecked_document_is_excluded_from_retrieval(seeded_db, monkeypatch) -> None:
    db, collection_id = seeded_db

    captured_document_ids: list[list[str] | None] = []
    original_load = retrieval_module._load_candidate_chunks

    def _spy(db_arg, document_ids):
        captured_document_ids.append(document_ids)
        return original_load(db_arg, document_ids)
    monkeypatch.setattr(retrieval_module, "_load_candidate_chunks", _spy)

    request = ChatRequest(question="What is ZigBee?", document_ids=["unit-1"], collection_id=collection_id)
    result = asyncio.run(ask(request, db=db))

    assert all(doc_id != "unit-2" for ids in captured_document_ids if ids for doc_id in ids)
    assert all(citation["document_id"] == "unit-1" for citation in result["citations"])


def test_empty_selection_inside_a_collection_is_a_controlled_error(seeded_db) -> None:
    db, collection_id = seeded_db
    request = ChatRequest(question="What is ZigBee?", document_ids=[], collection_id=collection_id)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(ask(request, db=db))
    assert exc_info.value.status_code == 400
    assert "no documents selected" in exc_info.value.detail.lower()


def test_documents_outside_the_collection_are_never_used_even_if_submitted(seeded_db) -> None:
    """SECURITY: the frontend's document_ids are never trusted blindly -- a
    document outside the collection's real membership must be filtered out
    server-side (see SECURITY / DATA ISOLATION)."""
    db, collection_id = seeded_db
    request = ChatRequest(question="What is ZigBee?", document_ids=["unit-1", "outside"], collection_id=collection_id)

    result = asyncio.run(ask(request, db=db))

    assert all(citation["document_id"] != "outside" for citation in result["citations"])


def test_submitting_only_out_of_collection_ids_is_treated_as_empty_selection(seeded_db) -> None:
    db, collection_id = seeded_db
    request = ChatRequest(question="What is ZigBee?", document_ids=["outside"], collection_id=collection_id)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(ask(request, db=db))
    assert exc_info.value.status_code == 400


def test_unknown_collection_id_returns_404(seeded_db) -> None:
    db, _ = seeded_db
    request = ChatRequest(question="What is ZigBee?", document_ids=["unit-1"], collection_id="does-not-exist")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(ask(request, db=db))
    assert exc_info.value.status_code == 404


def test_multi_document_collection_answer_cites_each_contributing_document(seeded_db, monkeypatch) -> None:
    async def _fake_generate(messages, **kwargs):
        return json.dumps({"answer": "Unit 1.pdf covers ZigBee [1] while Unit 2.pdf covers Bluetooth [2].", "source_ids": [1, 2]})
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    db, collection_id = seeded_db
    request = ChatRequest(question="Compare ZigBee and Bluetooth.", document_ids=["unit-1", "unit-2"], collection_id=collection_id)

    result = asyncio.run(ask(request, db=db))

    cited_documents = {citation["document_id"] for citation in result["citations"]}
    assert cited_documents == {"unit-1", "unit-2"}


def test_chat_without_collection_id_keeps_pre_group5_behavior(seeded_db) -> None:
    """REGRESSION: an ad-hoc multi-select with no collection_id must behave
    exactly as before -- no server-side membership filtering applied."""
    db, _ = seeded_db
    request = ChatRequest(question="What is ZigBee?", document_ids=["unit-1", "outside"])

    result = asyncio.run(ask(request, db=db))

    assert result["citations"]  # ad-hoc selection is used as-is, unfiltered
