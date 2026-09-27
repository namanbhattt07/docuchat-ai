import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Chunk, Document
from app.services.ollama import OllamaUnavailable
import app.services.retrieval as retrieval_module
from app.services.retrieval import is_broad_query, lexical_score, retrieve_overview, rewrite_standalone_question


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed_document(session, pages: int, chunks_per_page: int) -> str:
    document = Document(filename="test.pdf", page_count=pages, status="ready")
    session.add(document)
    session.flush()
    for page in range(1, pages + 1):
        for index in range(chunks_per_page):
            session.add(Chunk(id=f"{document.id}-{page}-{index}", document_id=document.id, page_number=page, content=f"Page {page} chunk {index} discusses overfitting and regularization."))
    session.commit()
    return document.id


@pytest.mark.parametrize(
    "question",
    [
        "brief me about this pdf",
        "Can you summarize this document?",
        "give me an overview",
        "what is this document about",
        "tell me all the topics covered in this pdf",
    ],
)
def test_is_broad_query_detects_summary_style_questions(question: str) -> None:
    assert is_broad_query(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is overfitting?",
        "What ratio of training and testing data did we use?",
        "Why did we use XGBoost?",
    ],
)
def test_is_broad_query_ignores_specific_questions(question: str) -> None:
    assert is_broad_query(question) is False


def test_lexical_score_zero_for_unrelated_content() -> None:
    assert lexical_score("what is overfitting", "a completely unrelated sentence about cats") == 0.0


def test_lexical_score_zero_for_empty_question() -> None:
    assert lexical_score("", "any content here") == 0.0


def test_lexical_score_rewards_exact_phrase_matches() -> None:
    loose = lexical_score("training test split ratio", "the dataset covers a training ratio")
    exact_phrase = lexical_score("training test split ratio", "we used an 85/15 training test split ratio for validation")
    assert exact_phrase > loose


def test_retrieve_overview_spans_every_page(db_session) -> None:
    document_id = _seed_document(db_session, pages=9, chunks_per_page=150)

    sources = retrieve_overview(db_session, [document_id], limit=9)

    assert {source["page_number"] for source in sources} == set(range(1, 10))


def test_retrieve_overview_returns_empty_for_unknown_document(db_session) -> None:
    assert retrieve_overview(db_session, ["missing-id"]) == []


def test_retrieve_overview_respects_limit(db_session) -> None:
    document_id = _seed_document(db_session, pages=3, chunks_per_page=5)

    sources = retrieve_overview(db_session, [document_id], limit=2)

    assert len(sources) == 2


def test_rewrite_standalone_question_uses_generate(monkeypatch) -> None:
    captured = {}

    async def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "What are the types of sensors?"

    monkeypatch.setattr(retrieval_module, "generate", fake_generate)

    result = asyncio.run(rewrite_standalone_question("PRIOR CONVERSATION...\n\n", "what are its type"))

    assert result == "What are the types of sensors?"
    assert captured["kwargs"].get("think") is False
    assert "what are its type" in captured["messages"][0]["content"]


def test_rewrite_standalone_question_falls_back_when_ollama_is_down(monkeypatch) -> None:
    async def failing_generate(messages, **kwargs):
        raise OllamaUnavailable("down")

    monkeypatch.setattr(retrieval_module, "generate", failing_generate)

    result = asyncio.run(rewrite_standalone_question("history", "what are its type"))

    assert result == "what are its type"


def test_rewrite_standalone_question_strips_think_block(monkeypatch) -> None:
    async def fake_generate(messages, **kwargs):
        return "<think>reasoning about pronouns</think>What is the frequency band of bluetooth?"

    monkeypatch.setattr(retrieval_module, "generate", fake_generate)

    result = asyncio.run(rewrite_standalone_question("history", "and its frequency band?"))

    assert result == "What is the frequency band of bluetooth?"
