import asyncio
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.chat as chat_module
from app.api.v1.chat import AbstractRequest, abstract
from app.db import Base
from app.models import Chunk, Conversation, Document, Message
from app.services.ollama import OllamaUnavailable


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed_document(session, pages=6):
    document = Document(filename="test.pdf", status="ready")
    session.add(document)
    session.flush()
    for page in range(1, pages + 1):
        session.add(Chunk(id=f"{document.id}-{page}-0", document_id=document.id, page_number=page, content=f"Page {page} discusses IoT concept number {page}."))
    session.commit()
    return document


def test_abstract_uses_overview_retrieval_for_document_wide_coverage(db_session, monkeypatch) -> None:
    document = _seed_document(db_session, pages=8)
    captured = {}

    async def fake_generate(messages, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return json.dumps({"answer": "# Abstract\n## Main Topic\nIoT concepts. [1]", "source_ids": [1, 2]})

    monkeypatch.setattr(chat_module, "generate", fake_generate)

    result = asyncio.run(abstract(AbstractRequest(document_ids=[document.id]), db=db_session))

    assert "# Abstract" in result["answer"]
    assert len(result["citations"]) == 2
    # OVERVIEW retrieval must span the whole document, not just the first N chunks.
    passages_section = captured["prompt"].split("PASSAGES:")[1].lower()
    pages_mentioned = {int(page) for page in __import__("re").findall(r"page (\d+)", passages_section)}
    assert len(pages_mentioned) > 1
    assert max(pages_mentioned) >= 6


def test_abstract_structured_outline_present_in_instructions(db_session, monkeypatch) -> None:
    document = _seed_document(db_session, pages=3)
    captured = {}

    async def fake_generate(messages, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return json.dumps({"answer": "# Abstract\n## Main Topic\ntext [1]", "source_ids": [1]})

    monkeypatch.setattr(chat_module, "generate", fake_generate)

    asyncio.run(abstract(AbstractRequest(document_ids=[document.id]), db=db_session))

    assert "## Key Ideas" in captured["prompt"]
    assert "## Important Sections" in captured["prompt"]
    assert "## Key Terms / Concepts" in captured["prompt"]


def test_abstract_raises_400_when_no_sources(db_session) -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(abstract(AbstractRequest(document_ids=["missing-doc"]), db=db_session))
    assert exc_info.value.status_code == 400


def test_abstract_returns_503_when_ollama_is_down(db_session, monkeypatch) -> None:
    document = _seed_document(db_session, pages=3)

    async def failing_generate(messages, **kwargs):
        raise OllamaUnavailable("down")

    monkeypatch.setattr(chat_module, "generate", failing_generate)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(abstract(AbstractRequest(document_ids=[document.id]), db=db_session))
    assert exc_info.value.status_code == 503


def test_failed_abstract_with_no_sources_leaves_no_orphan_conversation(db_session) -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(abstract(AbstractRequest(document_ids=["missing-doc"]), db=db_session))
    assert exc_info.value.status_code == 400

    assert db_session.query(Conversation).count() == 0
    assert db_session.query(Message).count() == 0


def test_failed_abstract_when_ollama_is_down_leaves_no_orphan_conversation(db_session, monkeypatch) -> None:
    document = _seed_document(db_session, pages=3)

    async def failing_generate(messages, **kwargs):
        raise OllamaUnavailable("down")

    monkeypatch.setattr(chat_module, "generate", failing_generate)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(abstract(AbstractRequest(document_ids=[document.id]), db=db_session))
    assert exc_info.value.status_code == 503

    assert db_session.query(Conversation).count() == 0
    assert db_session.query(Message).count() == 0


def test_abstract_persists_conversation_messages_with_citations(db_session, monkeypatch) -> None:
    document = _seed_document(db_session, pages=3)

    async def fake_generate(messages, **kwargs):
        return json.dumps({"answer": "Abstract text [1]", "source_ids": [1]})

    monkeypatch.setattr(chat_module, "generate", fake_generate)

    result = asyncio.run(abstract(AbstractRequest(document_ids=[document.id]), db=db_session))

    stored = db_session.query(Message).filter(Message.conversation_id == result["conversation_id"]).order_by(Message.created_at).all()
    assert [message.role for message in stored] == ["user", "assistant"]
    assert json.loads(stored[1].citations_json)[0]["document_id"] == document.id


def test_abstract_reuses_existing_conversation(db_session, monkeypatch) -> None:
    document = _seed_document(db_session, pages=3)

    async def fake_generate(messages, **kwargs):
        return json.dumps({"answer": "Abstract text [1]", "source_ids": [1]})

    monkeypatch.setattr(chat_module, "generate", fake_generate)

    first = asyncio.run(abstract(AbstractRequest(document_ids=[document.id]), db=db_session))
    second = asyncio.run(abstract(AbstractRequest(document_ids=[document.id], conversation_id=first["conversation_id"]), db=db_session))

    assert second["conversation_id"] == first["conversation_id"]
    stored = db_session.query(Message).filter(Message.conversation_id == first["conversation_id"]).all()
    assert len(stored) == 4
