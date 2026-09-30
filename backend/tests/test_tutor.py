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
from app.models import Chunk, Collection, CollectionDocument, Conversation, Document, Message
from app.services import tutor
from app.services.learning import explanatory_plan
from app.services.retrieval_types import RetrievalMode, RetrievalPlan
from app.services.tutor import TUTOR_ACTIONS, TutorAction, TutorContextError, TutorSession

# Group 6 TUTOR MODE + TUTOR ACTIONS: prompt construction, per-action
# instructions, explicit session state, selection handling, follow-up context,
# and the grounding/citation contract on tutor replies.


def _payload(**overrides) -> str:
    payload = {
        "topic": "ZigBee",
        "document_points": "ZigBee is a low-power wireless protocol for mesh networks. [1]",
        "tutor_explanation": "Think of it as a way for small devices to talk while sipping battery.",
        "example_kind": "example",
        "example": "A smart bulb relaying a message from a door sensor to the hub.",
        "check_question": "",
        "source_ids": [1],
    }
    payload.update(overrides)
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Prompt construction + actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", [action for action in TutorAction if action != TutorAction.CHECK_ANSWER])
def test_every_ui_action_maps_to_its_controlled_instruction(action) -> None:
    session = TutorSession(question=TUTOR_ACTIONS[action].label, action=action, current_topic="MQTT")
    prompt = tutor.build_tutor_prompt(session, "SOURCE 1 | a.pdf | page 1\nMQTT is a publish/subscribe protocol.", max_sources=4)

    assert TUTOR_ACTIONS[action].instruction in prompt
    assert f"TUTOR ACTION -- {TUTOR_ACTIONS[action].label}" in prompt


def test_required_action_instructions_match_the_brief() -> None:
    assert "simple language and a small example" in TUTOR_ACTIONS[TutorAction.EXPLAIN_SIMPLY].instruction
    assert "mechanism, assumptions and important details" in TUTOR_ACTIONS[TutorAction.EXPLAIN_DEEPLY].instruction
    assert "Do not immediately explain the answer" in TUTOR_ACTIONS[TutorAction.QUIZ_ME].instruction
    assert "without revealing the full answer" in TUTOR_ACTIONS[TutorAction.GIVE_HINT].instruction
    assert "prerequisites" in TUTOR_ACTIONS[TutorAction.TEACH_FROM_BEGINNING].instruction
    assert TUTOR_ACTIONS[TutorAction.GIVE_ANALOGY].example_kind == "analogy"
    assert TUTOR_ACTIONS[TutorAction.QUIZ_ME].requires_question
    assert not TUTOR_ACTIONS[TutorAction.QUIZ_ME].reveals_answer


def test_prompt_carries_explicit_session_state() -> None:
    session = TutorSession(
        question="Give example", action=TutorAction.GIVE_EXAMPLE, current_topic="MQTT broker",
        selected_text="The broker routes messages.", selected_page=5, pending_question="What does a broker do?",
        recent_turns="PRIOR CONVERSATION (context only, not evidence -- do not cite this section):\nUSER: hi\n\n",
    )
    prompt = tutor.build_tutor_prompt(session, "SOURCE 1 | a.pdf | user-selected passage\nThe broker routes messages.", max_sources=4)

    assert "Current topic: MQTT broker" in prompt
    assert "selected a passage in the PDF (page 5)" in prompt and "SOURCE 1" in prompt
    assert "PENDING QUESTION you asked the student: \"What does a broker do?\"" in prompt
    assert prompt.startswith("PRIOR CONVERSATION")
    assert "STUDENT: Give example" in prompt


def test_tutor_persona_is_a_teacher_and_grounding_rule_follows_mode() -> None:
    stick = tutor.build_tutor_system_message("document")
    free = tutor.build_tutor_system_message("free")

    assert "teaching assistant" in stick and "step by step" in stick
    assert "Stick to Document" in stick and "must not introduce facts" in stick
    assert "Go Freely" in free and "general knowledge" in free


def test_actions_on_a_topic_must_check_the_passages_cover_it_but_typed_questions_do_not() -> None:
    action = TutorSession(question="Explain simply", action=TutorAction.EXPLAIN_SIMPLY, current_topic="MQTT")
    typed = TutorSession(question="Now explain Bluetooth", current_topic="MQTT")
    assert "actually discuss \"MQTT\"" in tutor.build_tutor_prompt(action, "", max_sources=4)
    assert "doesn't cover it" in tutor.build_tutor_prompt(action, "", max_sources=4)
    assert "actually discuss" not in tutor.build_tutor_prompt(typed, "", max_sources=4)
    free = TutorSession(question="Explain simply", action=TutorAction.EXPLAIN_SIMPLY, current_topic="MQTT", grounding_mode="free")
    assert "general knowledge in tutor_explanation" in tutor.build_tutor_prompt(free, "", max_sources=4)


def test_format_instruction_is_layered_onto_the_tutor_explanation() -> None:
    session = TutorSession(question="Compare", action=TutorAction.COMPARE, current_topic="MQTT")
    prompt = tutor.build_tutor_prompt(session, "", max_sources=4, format_instruction="Format the answer as a table.")
    assert "PRESENTATION (applies to tutor_explanation): Format the answer as a table." in prompt


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def test_action_with_topic_searches_for_the_topic_not_the_button_label() -> None:
    retrieval = TutorSession(question="Why?", action=TutorAction.WHY, current_topic="MQTT QoS").retrieval()
    assert retrieval.query == "MQTT QoS" and not retrieval.needs_rewrite and not retrieval.document_wide


def test_action_without_topic_teaches_from_the_whole_document() -> None:
    retrieval = TutorSession(question="Teach from beginning", action=TutorAction.TEACH_FROM_BEGINNING).retrieval()
    assert retrieval.document_wide and retrieval.query is None


def test_typed_follow_up_goes_through_the_existing_rewrite_with_the_topic_in_context() -> None:
    session = TutorSession(question="why does it need one?", current_topic="MQTT broker", recent_turns="PRIOR:\n")
    assert session.retrieval().needs_rewrite
    assert "CURRENT TUTOR TOPIC: MQTT broker" in session.history_for_rewrite()


def test_hint_and_check_answer_search_for_the_pending_question() -> None:
    hint = TutorSession(question="Give hint", action=TutorAction.GIVE_HINT, current_topic="MQTT", pending_question="What port does MQTT use?")
    check = TutorSession(question="1883", action=TutorAction.CHECK_ANSWER, current_topic="MQTT", pending_question="What port does MQTT use?")
    assert "What port does MQTT use?" in hint.retrieval().query
    assert "What port does MQTT use?" in check.retrieval().query


def test_selection_takes_priority_and_blank_selection_is_ignored() -> None:
    assert TutorSession(question="Explain simply", selected_text="   \n ").selected_text is None
    session = TutorSession(question="Explain simply", action=TutorAction.EXPLAIN_SIMPLY, selected_text="QoS 2 guarantees exactly-once delivery.")
    assert session.has_context
    assert not session.retrieval().document_wide  # the selection, not the whole document


@pytest.mark.parametrize("action", [TutorAction.WHY, TutorAction.COMPARE, TutorAction.GIVE_HINT])
def test_context_dependent_actions_refuse_without_a_topic_or_selection(action) -> None:
    with pytest.raises(TutorContextError):
        TutorSession(question=TUTOR_ACTIONS[action].label, action=action).validate()


def test_check_answer_requires_an_open_question() -> None:
    with pytest.raises(TutorContextError):
        TutorSession(question="42", action=TutorAction.CHECK_ANSWER, current_topic="MQTT").validate()


def test_explanatory_plan_never_lets_the_tutor_answer_with_a_location_list() -> None:
    plan = RetrievalPlan(mode=RetrievalMode.LOCATION, query="ZigBee", location_topic="ZigBee", top_k=6)
    adapted = explanatory_plan(plan, document_wide=False, overview_top_k=10, default_top_k=8)
    assert adapted.mode == RetrievalMode.FACT and adapted.top_k == 8 and adapted.location_topic is None


def test_explanatory_plan_keeps_selection_even_for_document_wide_actions() -> None:
    plan = RetrievalPlan(mode=RetrievalMode.SELECTION, query="x", selected_text="x")
    assert explanatory_plan(plan, document_wide=True, overview_top_k=10, default_top_k=8).mode == RetrievalMode.SELECTION


# ---------------------------------------------------------------------------
# Response parsing / grounding contract
# ---------------------------------------------------------------------------


def test_reply_separates_document_tutor_and_example_content() -> None:
    session = TutorSession(question="Explain simply", action=TutorAction.EXPLAIN_SIMPLY, current_topic="ZigBee")
    raw = _payload(tutor_explanation="It is like whispering between neighbours [1].")
    parsed = tutor.parse_tutor_response(raw, session, source_count=2, max_sources=4)

    kinds = [section.kind for section in parsed.sections]
    assert kinds == ["document", "tutor", "example"]
    assert "[1]" in parsed.sections[0].content
    assert "[1]" not in parsed.sections[1].content  # tutor explanation is never cited as document evidence
    assert "not taken from the document" in parsed.sections[2].note
    assert parsed.source_ids == [1]


def test_stick_to_document_refuses_a_reply_with_no_verified_evidence() -> None:
    session = TutorSession(question="Explain simply", action=TutorAction.EXPLAIN_SIMPLY, current_topic="ZigBee")
    parsed = tutor.parse_tutor_response(_payload(document_points="", source_ids=[7]), session, source_count=2, max_sources=4)

    assert parsed.insufficient_evidence
    assert parsed.source_ids == []
    assert [section.kind for section in parsed.sections] == ["notice"]
    assert "Go Freely" in parsed.sections[0].content


def test_go_freely_keeps_the_tutor_explanation_but_labels_it_general_knowledge() -> None:
    session = TutorSession(question="Explain simply", action=TutorAction.EXPLAIN_SIMPLY, current_topic="ZigBee", grounding_mode="free")
    parsed = tutor.parse_tutor_response(_payload(document_points="", source_ids=[]), session, source_count=2, max_sources=4)

    assert parsed.sections[0].kind == "notice" and "don't cover this" in parsed.sections[0].content
    tutor_section = next(section for section in parsed.sections if section.kind == "tutor")
    assert "general knowledge" in tutor_section.note


def test_quiz_me_withholds_the_answer_and_waits_for_the_student() -> None:
    session = TutorSession(question="Quiz me", action=TutorAction.QUIZ_ME, current_topic="ZigBee")
    parsed = tutor.parse_tutor_response(_payload(check_question="What kind of network topology does ZigBee use?"), session, 2, 4)

    assert "document" not in [section.kind for section in parsed.sections]
    assert parsed.sections[-1].kind == "question"
    assert parsed.awaiting_answer and parsed.metadata()["pending_question"].startswith("What kind")
    assert parsed.source_ids == [1]  # the question is still traceable to its passage


def test_quiz_me_without_a_question_fails_safely() -> None:
    session = TutorSession(question="Quiz me", action=TutorAction.QUIZ_ME, current_topic="ZigBee")
    parsed = tutor.parse_tutor_response(_payload(check_question=""), session, 2, 4)
    assert not parsed.awaiting_answer and parsed.sections[0].kind == "notice"


def test_malformed_output_with_citation_markers_is_still_treated_as_document_evidence() -> None:
    session = TutorSession(question="What is ZigBee?", current_topic=None)
    parsed = tutor.parse_tutor_response("ZigBee is a mesh protocol [2].", session, source_count=2, max_sources=4)
    assert parsed.source_ids == [2] and parsed.sections[0].kind == "document"


def test_malformed_uncited_output_fails_safely_under_stick_to_document() -> None:
    session = TutorSession(question="What is ZigBee?")
    parsed = tutor.parse_tutor_response("Some unverifiable prose.", session, source_count=2, max_sources=4)
    assert parsed.insufficient_evidence and parsed.sections[0].kind == "notice"


def test_out_of_range_citation_markers_are_removed_from_document_text() -> None:
    session = TutorSession(question="What is ZigBee?")
    parsed = tutor.parse_tutor_response(_payload(document_points="Fact one [1]. Fact nine [9].", source_ids=[1, 9]), session, 2, 4)
    assert "[9]" not in parsed.sections[0].content and parsed.source_ids == [1]


def test_actions_keep_the_current_topic_while_typed_questions_can_rename_it() -> None:
    action = TutorSession(question="Why?", action=TutorAction.WHY, current_topic="ZigBee")
    typed = TutorSession(question="Now explain Bluetooth", current_topic="ZigBee")
    assert tutor.parse_tutor_response(_payload(topic="Something else"), action, 2, 4).topic == "ZigBee"
    assert tutor.parse_tutor_response(_payload(topic="Bluetooth"), typed, 2, 4).topic == "Bluetooth"


def test_markdown_rendering_labels_every_section() -> None:
    session = TutorSession(question="Explain simply", action=TutorAction.EXPLAIN_SIMPLY, current_topic="ZigBee")
    markdown = tutor.parse_tutor_response(_payload(check_question="Can you name one ZigBee device?"), session, 2, 4).to_markdown()
    assert "### From the document" in markdown
    assert "### Tutor's explanation" in markdown
    assert "### Example" in markdown
    assert "### Your turn" in markdown


# ---------------------------------------------------------------------------
# Full /chat wiring (mode="tutor")
# ---------------------------------------------------------------------------


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

    async def _no_rewrite(history, question):
        return question
    monkeypatch.setattr(chat_module, "rewrite_standalone_question", _no_rewrite)

    db_session.add_all([
        Document(id="doc-1", filename="protocols.pdf", status="ready"),
        Document(id="doc-2", filename="other.pdf", status="ready"),
        Chunk(id="doc-1-1-0", document_id="doc-1", page_number=1, content="ZigBee is a low-power wireless protocol used for mesh networks.", start_offset=0, end_offset=60),
        Chunk(id="doc-1-2-0", document_id="doc-1", page_number=2, content="ZigBee devices route messages for each other across the mesh.", start_offset=0, end_offset=60),
        Chunk(id="doc-2-1-0", document_id="doc-2", page_number=1, content="Photosynthesis converts light into chemical energy in ZigBee leaves.", start_offset=0, end_offset=60),
    ])
    db_session.commit()
    return db_session


def _capture_generate(monkeypatch, raw: str) -> list:
    calls: list = []

    async def _fake_generate(messages, **kwargs):
        calls.append(messages)
        return raw
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    return calls


def test_tutor_action_runs_through_the_shared_chat_pipeline(seeded_db, monkeypatch) -> None:
    calls = _capture_generate(monkeypatch, _payload())
    request = ChatRequest(question="Explain simply", document_ids=["doc-1"], mode="tutor", tutor_action="explain_simply", current_topic="ZigBee")

    result = asyncio.run(ask(request, db=seeded_db))

    assert result["mode"] == "tutor"
    assert result["tutor"]["topic"] == "ZigBee" and result["tutor"]["action"] == "explain_simply"
    assert [section["kind"] for section in result["tutor"]["sections"]] == ["document", "tutor", "example"]
    assert result["citations"] and result["citations"][0]["document_id"] == "doc-1"  # existing citation shape
    system, user = calls[0]
    assert "DocuChat Tutor" in system["content"]
    assert "simple language and a small example" in user["content"]
    persisted = seeded_db.query(Message).filter(Message.role == "user").one()
    assert persisted.content == "[Tutor · Explain simply] ZigBee"
    assert seeded_db.get(Message, result["message_id"]).content.startswith("### From the document")


def test_tutor_follow_ups_keep_the_topic_across_turns(seeded_db, monkeypatch) -> None:
    """"Explain this simply" -> "Why?" -> "Give me an example": each action
    reuses the topic the previous tutor turn established, and the prior turns
    ride along as (non-citable) history."""
    calls = _capture_generate(monkeypatch, _payload())
    first = asyncio.run(ask(ChatRequest(question="Explain ZigBee simply", document_ids=["doc-1"], mode="tutor"), db=seeded_db))
    topic = first["tutor"]["topic"]

    async def _must_not_rewrite(history, question):
        raise AssertionError("action follow-ups should search the topic, not call the rewrite model")
    monkeypatch.setattr(chat_module, "rewrite_standalone_question", _must_not_rewrite)

    for action in ("why", "give_example"):
        result = asyncio.run(ask(ChatRequest(
            question=TUTOR_ACTIONS[TutorAction(action)].label, document_ids=["doc-1"], mode="tutor",
            tutor_action=action, current_topic=topic, conversation_id=first["conversation_id"],
        ), db=seeded_db))
        assert result["tutor"]["topic"] == "ZigBee"
        assert result["conversation_id"] == first["conversation_id"]

    last_prompt = calls[-1][1]["content"]
    assert "Current topic: ZigBee" in last_prompt
    assert "PRIOR CONVERSATION" in last_prompt and "[Tutor · Why?] ZigBee" in last_prompt


def test_tutor_quiz_then_answer_round_trip(seeded_db, monkeypatch) -> None:
    _capture_generate(monkeypatch, _payload(check_question="What topology does ZigBee use?"))
    quiz = asyncio.run(ask(ChatRequest(question="Quiz me", document_ids=["doc-1"], mode="tutor", tutor_action="quiz_me", current_topic="ZigBee"), db=seeded_db))
    assert quiz["tutor"]["awaiting_answer"]

    calls = _capture_generate(monkeypatch, _payload(tutor_explanation="Yes -- mesh is right!"))
    answer = asyncio.run(ask(ChatRequest(
        question="A mesh", document_ids=["doc-1"], mode="tutor", tutor_action="check_answer", current_topic="ZigBee",
        pending_question=quiz["tutor"]["pending_question"], conversation_id=quiz["conversation_id"],
    ), db=seeded_db))

    assert "PENDING QUESTION you asked the student: \"What topology does ZigBee use?\"" in calls[0][1]["content"]
    assert answer["tutor"]["sections"][1]["title"] == "Feedback on your answer"


def test_tutor_uses_the_selected_passage_as_primary_context(seeded_db, monkeypatch) -> None:
    calls = _capture_generate(monkeypatch, _payload())
    request = ChatRequest(
        question="Explain deeply", document_ids=["doc-1"], mode="tutor", tutor_action="explain_deeply",
        selected_text="ZigBee devices route messages for each other.", selection_page=2,
    )

    result = asyncio.run(ask(request, db=seeded_db))

    prompt = calls[0][1]["content"]
    assert "SOURCE 1 | protocols.pdf | user-selected passage\nZigBee devices route messages for each other." in prompt
    assert "(page 2)" in prompt
    assert result["citations"][0]["page_number"] == 2  # the selection's own page, via the existing selection retrieval


def test_context_dependent_action_is_rejected_before_creating_a_conversation(seeded_db, monkeypatch) -> None:
    _capture_generate(monkeypatch, _payload())
    with pytest.raises(HTTPException) as error:
        asyncio.run(ask(ChatRequest(question="Why?", document_ids=["doc-1"], mode="tutor", tutor_action="why"), db=seeded_db))
    assert error.value.status_code == 400
    assert seeded_db.query(Conversation).count() == 0


def test_blank_selection_with_no_topic_is_rejected_for_compare(seeded_db, monkeypatch) -> None:
    _capture_generate(monkeypatch, _payload())
    with pytest.raises(HTTPException) as error:
        asyncio.run(ask(ChatRequest(question="Compare", document_ids=["doc-1"], mode="tutor", tutor_action="compare", selected_text="   "), db=seeded_db))
    assert "needs something to work on" in error.value.detail


def test_teach_from_beginning_without_a_topic_uses_document_wide_coverage(seeded_db, monkeypatch) -> None:
    calls = _capture_generate(monkeypatch, _payload(topic="ZigBee basics"))
    result = asyncio.run(ask(ChatRequest(question="Teach from beginning", document_ids=["doc-1"], mode="tutor", tutor_action="teach_from_beginning"), db=seeded_db))
    prompt = calls[0][1]["content"]
    assert "page 1" in prompt and "page 2" in prompt  # overview sampled both pages
    assert result["tutor"]["topic"] == "ZigBee basics"


def test_tutor_respects_collection_scope(seeded_db, monkeypatch) -> None:
    calls = _capture_generate(monkeypatch, _payload())
    seeded_db.add(Collection(id="col-1", name="Networking"))
    seeded_db.add(CollectionDocument(collection_id="col-1", document_id="doc-1"))
    seeded_db.commit()

    asyncio.run(ask(ChatRequest(
        question="Explain simply", document_ids=["doc-1", "doc-2"], collection_id="col-1",
        mode="tutor", tutor_action="explain_simply", current_topic="ZigBee",
    ), db=seeded_db))
    assert "Photosynthesis" not in calls[0][1]["content"]  # doc-2 isn't in the collection

    with pytest.raises(HTTPException) as error:
        asyncio.run(ask(ChatRequest(
            question="Explain simply", document_ids=["doc-2"], collection_id="col-1",
            mode="tutor", tutor_action="explain_simply", current_topic="ZigBee",
        ), db=seeded_db))
    assert error.value.status_code == 400


def test_tutor_stick_to_document_fails_safely_end_to_end(seeded_db, monkeypatch) -> None:
    _capture_generate(monkeypatch, _payload(document_points="", source_ids=[]))
    result = asyncio.run(ask(ChatRequest(question="Explain simply", document_ids=["doc-1"], mode="tutor", tutor_action="explain_simply", current_topic="ZigBee"), db=seeded_db))
    assert result["citations"] == []
    assert result["tutor"]["insufficient_evidence"]
    assert "couldn't find enough" in result["answer"]


def test_tutor_go_freely_end_to_end_keeps_document_and_general_content_apart(seeded_db, monkeypatch) -> None:
    calls = _capture_generate(monkeypatch, _payload())
    result = asyncio.run(ask(ChatRequest(
        question="Explain deeply", document_ids=["doc-1"], mode="tutor", tutor_action="explain_deeply",
        current_topic="ZigBee", grounding_mode="free",
    ), db=seeded_db))
    assert "Go Freely" in calls[0][0]["content"]
    tutor_section = next(section for section in result["tutor"]["sections"] if section["kind"] == "tutor")
    assert "general knowledge" in tutor_section["note"]
    assert "[1]" not in tutor_section["content"]


def test_plain_chat_mode_is_unchanged_by_tutor_fields(seeded_db, monkeypatch) -> None:
    _capture_generate(monkeypatch, json.dumps({"answer": "ZigBee is a mesh protocol. [1]", "source_ids": [1]}))
    result = asyncio.run(ask(ChatRequest(question="What is ZigBee?", document_ids=["doc-1"]), db=seeded_db))
    assert "tutor" not in result and "mode" not in result
    assert result["answer"] == "ZigBee is a mesh protocol. [1]"
