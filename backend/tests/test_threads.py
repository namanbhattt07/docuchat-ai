import asyncio
import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.chat as chat_module
import app.services.retrieval as retrieval_module
from app.api.v1.chat import HISTORY_TURNS, ChatRequest, _recent_history, ask, get_conversation
from app.db import Base
from app.models import Chunk, Conversation, Document, Message

# Group 5: THREADS / FOLLOW-UPS (Reply affordance + bounded history). See
# QUERY REWRITE FIX / HISTORY EFFICIENCY / THREAD TEST in the brief.


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed_turn(db, conversation_id: str, index: int, base_time: datetime) -> tuple[Message, Message]:
    user = Message(
        id=f"user-{index}", conversation_id=conversation_id, role="user",
        content=f"Question number {index}", created_at=base_time + timedelta(seconds=index * 2),
    )
    assistant = Message(
        id=f"assistant-{index}", conversation_id=conversation_id, role="assistant",
        content=f"Answer number {index}", created_at=base_time + timedelta(seconds=index * 2 + 1),
    )
    db.add_all([user, assistant])
    return user, assistant


def test_recent_history_bounds_to_the_most_recent_turns_only(db_session) -> None:
    conversation = Conversation(id="conv-1", title="Test")
    db_session.add(conversation)
    base_time = datetime(2026, 1, 1)
    for index in range(HISTORY_TURNS + 5):
        _seed_turn(db_session, "conv-1", index, base_time)
    db_session.commit()

    history = _recent_history(db_session, "conv-1")

    assert "Question number 0" not in history  # the oldest turns fell outside the bounded window
    assert f"Question number {HISTORY_TURNS + 4}" in history  # the most recent turn is always included
    assert history.count("USER:") <= HISTORY_TURNS
    assert "do not cite this section" in history.lower()


def test_recent_history_pulls_in_a_reply_anchor_outside_the_normal_window(db_session) -> None:
    """Group 5 Reply: hitting "Reply" on an old turn must still surface that
    turn's context even though it has scrolled out of the bounded window."""
    conversation = Conversation(id="conv-1", title="Test")
    db_session.add(conversation)
    base_time = datetime(2026, 1, 1)
    _seed_turn(db_session, "conv-1", 0, base_time)  # the turn we'll reply to
    for index in range(1, HISTORY_TURNS + 5):
        _seed_turn(db_session, "conv-1", index, base_time)
    db_session.commit()

    history = _recent_history(db_session, "conv-1", anchor_message_id="assistant-0")

    assert "Answer number 0" in history
    assert f"Question number {HISTORY_TURNS + 4}" in history  # the normal recent window is preserved too


def test_recent_history_ignores_an_anchor_from_a_different_conversation(db_session) -> None:
    db_session.add_all([Conversation(id="conv-1", title="A"), Conversation(id="conv-2", title="B")])
    base_time = datetime(2026, 1, 1)
    _seed_turn(db_session, "conv-2", 0, base_time)  # belongs to the OTHER conversation
    db_session.commit()

    history = _recent_history(db_session, "conv-1", anchor_message_id="assistant-0")

    assert history == ""  # conv-1 has no messages of its own, and the foreign anchor must not leak in


def test_existing_messages_without_a_reply_target_remain_compatible(db_session) -> None:
    """REGRESSION: rows created before Group 5 have reply_to_message_id=NULL
    -- history-building must not choke on that."""
    conversation = Conversation(id="conv-1", title="Test")
    db_session.add(conversation)
    db_session.add(Message(id="m-1", conversation_id="conv-1", role="user", content="Old-style question"))
    db_session.add(Message(id="m-2", conversation_id="conv-1", role="assistant", content="Old-style answer"))
    db_session.commit()

    history = _recent_history(db_session, "conv-1")

    assert "Old-style question" in history


# ---------------------------------------------------------------------------
# Full /chat wiring: message ids + reply_to_message_id round trip
# ---------------------------------------------------------------------------


class _FakeCollection:
    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def seeded_db(db_session, monkeypatch):
    monkeypatch.setattr(chat_module, "get_collection", lambda: _FakeCollection())

    async def _fake_embed(texts):
        return [[0.0] * 4 for _ in texts]
    monkeypatch.setattr(retrieval_module, "embed", _fake_embed)

    async def _fake_generate(messages, **kwargs):
        return json.dumps({"answer": "ZigBee is a low-power wireless protocol. [1]", "source_ids": [1]})
    monkeypatch.setattr(chat_module, "generate", _fake_generate)

    # Follow-up turns have non-empty history, which would otherwise trigger a
    # real rewrite_standalone_question() call straight to the local Ollama
    # server -- keep these unit tests hermetic/fast by making it a passthrough
    # by default; tests that care about the rewrite prompt itself override
    # this individually (see test_reply_pulls_replied_to_turn_into_the_rewrite_context).
    async def _fake_rewrite(history, question):
        return question
    monkeypatch.setattr(chat_module, "rewrite_standalone_question", _fake_rewrite)

    db_session.add(Document(id="doc-1", filename="protocols.pdf", status="ready"))
    db_session.add(Chunk(id="doc-1-1-0", document_id="doc-1", page_number=1, content="ZigBee is a low-power wireless protocol used for mesh networks.", start_offset=0, end_offset=60))
    db_session.commit()
    return db_session


def test_ask_returns_a_message_id_for_the_reply_affordance(seeded_db) -> None:
    request = ChatRequest(question="What is ZigBee?", document_ids=["doc-1"])

    result = asyncio.run(ask(request, db=seeded_db))

    assert result["message_id"]
    assert seeded_db.get(Message, result["message_id"]) is not None


def test_reply_to_message_id_is_persisted_on_the_follow_up_user_message(seeded_db) -> None:
    first = asyncio.run(ask(ChatRequest(question="What is ZigBee?", document_ids=["doc-1"]), db=seeded_db))
    conversation_id = first["conversation_id"]
    first_assistant_id = first["message_id"]

    asyncio.run(ask(ChatRequest(
        question="Why is ZigBee useful?", document_ids=["doc-1"],
        conversation_id=conversation_id, reply_to_message_id=first_assistant_id,
    ), db=seeded_db))

    follow_up_user_message = seeded_db.query(Message).filter(
        Message.conversation_id == conversation_id, Message.role == "user", Message.content == "Why is ZigBee useful?",
    ).one()
    assert follow_up_user_message.reply_to_message_id == first_assistant_id


def test_get_conversation_exposes_reply_to_message_id(seeded_db) -> None:
    first = asyncio.run(ask(ChatRequest(question="What is ZigBee?", document_ids=["doc-1"]), db=seeded_db))
    asyncio.run(ask(ChatRequest(
        question="Why is ZigBee useful?", document_ids=["doc-1"],
        conversation_id=first["conversation_id"], reply_to_message_id=first["message_id"],
    ), db=seeded_db))

    payload = get_conversation(first["conversation_id"], db=seeded_db)

    follow_up = next(m for m in payload["messages"] if m["content"] == "Why is ZigBee useful?")
    assert follow_up["reply_to_message_id"] == first["message_id"]


def test_reply_pulls_replied_to_turn_into_the_rewrite_context(seeded_db, monkeypatch) -> None:
    """THREAD TEST: a comparison follow-up ("Compare it with Bluetooth")
    should have the replied-to turn available for reference resolution, even
    if the conversation moved on in between."""
    first = asyncio.run(ask(ChatRequest(question="What is ZigBee?", document_ids=["doc-1"]), db=seeded_db))

    captured_history = {}

    async def _capture_rewrite(history, question):
        captured_history["value"] = history
        return question
    monkeypatch.setattr(chat_module, "rewrite_standalone_question", _capture_rewrite)

    asyncio.run(ask(ChatRequest(
        question="Compare ZigBee with Bluetooth.", document_ids=["doc-1"],
        conversation_id=first["conversation_id"], reply_to_message_id=first["message_id"],
    ), db=seeded_db))

    assert "ZigBee is a low-power wireless protocol" in captured_history["value"]
