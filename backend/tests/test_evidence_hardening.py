import asyncio
import json

import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider
from fastapi import HTTPException
from sqlalchemy import func, select

import app.api.v1.chat as chat_module
import app.api.v1.learning as learning_module
from app.api.v1.chat import ChatRequest, ask
from app.api.v1.learning import QuestionGenerationRequest, generate_question_set
from app.models import Chunk, Conversation, Document, PageText
from app.services import calculation, evidence, extraction, templates
from app.services.extraction import parse_extraction_response
from app.services.providers import ProviderUnavailable, use_providers
from app.services.retrieval import retrieve_page
from app.services.retrieval_types import RetrievalMode, RetrievalPlan

# Group 8 -- failing closed. Every mode must answer "the document doesn't say"
# (or refuse) rather than turn unrelated or missing evidence into an answer. The
# cases here were each reproduced against the live app with the real model
# before being fixed; the model is faked so the check is on the code's own
# guards, not on how well a particular model behaves.

INSUFFICIENT = "I couldn't verify a source-grounded answer from the retrieved document passages."


class _NoVectorHits:
    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def library(db_session, monkeypatch):
    """Two ready documents, a blank page, an OCR-failed page, and one document
    of each other lifecycle state."""
    monkeypatch.setattr(chat_module, "get_collection", lambda: _NoVectorHits())
    monkeypatch.setattr(learning_module, "get_collection", lambda: _NoVectorHits())
    db_session.add_all([
        Document(id="iot", filename="iot.pdf", status="ready", page_count=4),
        Document(id="market", filename="market.pdf", status="ready", page_count=2),
        Document(id="busy", filename="busy.pdf", status="processing"),
        Document(id="broken", filename="broken.pdf", status="failed", status_detail="This PDF is password-protected. Remove the password, then upload it again."),
        Document(id="blank", filename="blank.pdf", status="empty"),
    ])
    chunks = [
        ("iot", 1, "Introduction", "This report surveys wireless protocols used in Internet of Things deployments and compares ZigBee, LoRa and BLE."),
        ("iot", 2, "Wireless Communication Protocols", "ZigBee operates in the 2.4 GHz frequency band and uses a mesh topology with data rates up to 250 kbps."),
        ("iot", 4, "Sensors", "The pilot used temperature sensors, humidity sensors and motion sensors. Each temperature sensor reports every 60 seconds."),
        ("market", 1, "Market Overview", "The smart home market grew steadily; voice assistants were the fastest growing device category."),
    ]
    for document_id, page, section, text in chunks:
        db_session.add(Chunk(id=f"{document_id}-{page}-0", document_id=document_id, page_number=page, content=text, section=section, start_offset=0, end_offset=len(text)))
    db_session.add_all([
        PageText(id="iot-3", document_id="iot", page_number=3, content="", status="empty", status_detail="Blank page."),
    ])
    db_session.commit()
    return db_session


def _ask(db, question, *, llm="", document_ids=("iot",), **extra):
    model = FakeLLMProvider(llm)
    with use_providers(llm=model, embedding=FakeEmbeddingProvider()):
        result = asyncio.run(ask(ChatRequest(question=question, document_ids=list(document_ids), **extra), db=db))
    return result, model


def _fabricating(**extra) -> str:
    """What a model that ignores the passages says: a confident answer that
    cites nothing that exists."""
    return json.dumps({"answer": "The capital of France is Paris. [9]", "source_ids": [9], **extra})


# ---------------------------------------------------------------------------
# The matrix: every retrieval mode, model answers from nowhere
# ---------------------------------------------------------------------------

MODE_QUESTIONS = [
    ("fact", "What frequency does ZigBee use?", ("iot",), {}),
    ("overview", "Give me a summary of this document", ("iot",), {}),
    ("section", "What does the Sensors section say?", ("iot",), {}),
    ("page", "What is on page 1?", ("iot",), {}),
    ("multi_document", "What do these documents say about ZigBee and voice assistants?", ("iot", "market"), {}),
    ("selection", "Explain the selected passage.", ("iot",), {"selected_text": "ZigBee operates in the 2.4 GHz frequency band.", "selection_page": 2, "selection_action": "explain"}),
]


@pytest.mark.parametrize("mode, question, documents, extra", MODE_QUESTIONS, ids=[case[0] for case in MODE_QUESTIONS])
def test_stick_to_document_never_shows_an_answer_that_cites_nothing_real(library, mode, question, documents, extra) -> None:
    result, model = _ask(library, question, llm=_fabricating(), document_ids=documents, **extra)

    assert model.calls, "this mode should have reached the model"
    assert result["answer"].endswith(INSUFFICIENT)  # selection mode prefixes its own lead-in
    assert "Paris" not in result["answer"]
    assert result["citations"] == []


@pytest.mark.parametrize("mode, question, documents, extra", MODE_QUESTIONS, ids=[case[0] for case in MODE_QUESTIONS])
def test_go_freely_keeps_general_knowledge_out_of_the_document_answer_and_uncited(library, mode, question, documents, extra) -> None:
    raw = _fabricating(additional_context="Paris has been France's capital for centuries. [1]")
    result, _ = _ask(library, question, llm=raw, document_ids=documents, grounding_mode="free", **extra)

    # The document part fails closed; the general knowledge survives only under its own heading, with no citation.
    assert "Paris has been France's capital for centuries." in result["answer"]
    assert "The capital of France is Paris." not in result["answer"]
    assert "[1]" not in result["answer"] and "[9]" not in result["answer"]
    assert result["citations"] == []


def test_stick_to_document_ignores_additional_context_entirely(library) -> None:
    result, _ = _ask(library, "What frequency does ZigBee use?", llm=_fabricating(additional_context="smuggled general knowledge"))

    assert "smuggled" not in result["answer"]


# ---------------------------------------------------------------------------
# PAGE: a page that isn't there is a definite answer, never a neighbour's text
# ---------------------------------------------------------------------------


def _no_model(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("no model call is needed to say a page doesn't exist")

    monkeypatch.setattr(chat_module, "generate", forbidden)


@pytest.mark.parametrize("question, expected", [
    ("What is discussed on page 99?", 'has 4 pages, so there is no page 99'),
    ("What does page 5 say?", 'has 4 pages, so there is no page 5'),  # the page just past the end must not borrow page 4
    ("Summarize page 0", "no page 0"),
])
def test_a_page_that_does_not_exist_is_reported_from_the_page_count(library, monkeypatch, question, expected) -> None:
    _no_model(monkeypatch)
    result, _ = _ask(library, question)

    assert expected in result["answer"]
    assert result["citations"] == []


def test_a_blank_page_is_reported_as_blank_not_answered_from_its_neighbours(library, monkeypatch) -> None:
    _no_model(monkeypatch)
    result, _ = _ask(library, "What is on page 3?")

    assert "Page 3 has no readable text" in result["answer"]
    assert result["citations"] == []


def test_a_page_ocr_could_not_read_says_why(library, monkeypatch) -> None:
    _no_model(monkeypatch)
    library.query(PageText).filter_by(id="iot-3").update({"status": "failed", "status_detail": "OCR failed on this page: engine crashed"})
    library.commit()

    result, _ = _ask(library, "What is on page 3?")

    assert "Page 3 could not be read" in result["answer"] and "engine crashed" in result["answer"]


def test_page_retrieval_returns_nothing_rather_than_neighbours_when_the_page_has_no_text(library) -> None:
    plan = RetrievalPlan(mode=RetrievalMode.PAGE, query="", document_ids=["iot"], page=3, top_k=8, diversity=False)

    assert retrieve_page(library, plan) == []  # pages 2 and 4 have text, but they are not page 3


def test_a_real_page_is_still_answered(library) -> None:
    good = json.dumps({"answer": "ZigBee uses 2.4 GHz. [1]", "source_ids": [1]})
    result, _ = _ask(library, "What is on page 2?", llm=good)

    assert result["citations"][0]["page_number"] == 2


# ---------------------------------------------------------------------------
# Empty scope: say what is wrong with the selected documents
# ---------------------------------------------------------------------------

EMPTY_SCOPE_MODES = [
    ("chat", {}),
    ("tutor", {"mode": "tutor"}),
    ("brainstorm", {"mode": "brainstorm"}),
    ("template", {"mode": "template", "template_id": "builtin:executive-summary"}),
]


@pytest.mark.parametrize("mode, extra", EMPTY_SCOPE_MODES, ids=[case[0] for case in EMPTY_SCOPE_MODES])
@pytest.mark.parametrize("document_id, expected", [
    ("busy", "still being processed"),
    ("broken", "password-protected"),
    ("blank", "no extractable text"),
])
def test_asking_about_a_document_that_isnt_usable_explains_why(library, mode, extra, document_id, expected) -> None:
    with pytest.raises(HTTPException) as error:
        _ask(library, "What does it say?", document_ids=(document_id,), **extra)

    assert error.value.status_code == 400
    assert expected in error.value.detail
    assert "Upload a text-based PDF" not in error.value.detail  # wrong advice for a document that exists


def test_a_rejected_request_leaves_no_empty_conversation_behind(library) -> None:
    with pytest.raises(HTTPException):
        _ask(library, "What does it say?", document_ids=("busy",))
    with pytest.raises(HTTPException):
        _ask(library, "What does it say?", llm=ProviderUnavailable("down"))  # a model outage after retrieval

    assert library.scalar(select(func.count()).select_from(Conversation)) == 0


def test_an_answered_question_still_stores_its_conversation(library) -> None:
    good = json.dumps({"answer": "ZigBee uses 2.4 GHz. [1]", "source_ids": [1]})
    result, _ = _ask(library, "What frequency does ZigBee use?", llm=good)

    conversation = library.get(Conversation, result["conversation_id"])
    assert conversation is not None and conversation.title == "What frequency does ZigBee use?"


# ---------------------------------------------------------------------------
# LOCATION: pages must be about the whole topic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question, pages", [
    ("Where does the document discuss ZigBee?", [1, 2]),  # named on the intro page as well as its own section
    ("Where does the document discuss temperature sensors?", [4]),
    ("Where does the document discuss the sensor accuracy?", []),  # 'accuracy' appears nowhere in this fixture
    ("Where does the document discuss quantum sensors?", []),  # 'sensors' is on page 4, but the topic is 'quantum sensors'
    ("Where does the document discuss blockchain voting?", []),
])
def test_location_answers_only_name_pages_that_cover_the_whole_topic(library, monkeypatch, question, pages) -> None:
    _no_model(monkeypatch)
    result, _ = _ask(library, question)

    assert sorted(citation["page_number"] for citation in result["citations"]) == pages
    if not pages:
        assert "couldn't find" in result["answer"]


# ---------------------------------------------------------------------------
# CALCULATION: a year is when, not how much
# ---------------------------------------------------------------------------

FINANCIALS = [{"content": "Revenue was 80000 dollars in 2022 and 95000 dollars in 2023. Total cost was 50000 dollars in 2022 and 52000 dollars in 2023."}]


@pytest.mark.parametrize("question, expected", [
    ("What is the percentage increase in revenue from 2022 to 2023?", 18.75),
    ("What is the total cost in 2022 and 2023?", 102000.0),
    ("What is the difference between revenue and cost?", 30000.0),
    ("How much more was revenue than cost?", 30000.0),
    ("What is the percentage increase from 1990 to 2010?", pytest.approx(1.005, abs=0.001)),  # no quantity named: the user's own numbers
    ("What is the difference between 2000 and 1500?", 500.0),
])
def test_years_are_never_used_as_operands(question, expected) -> None:
    outcome = calculation.compute(question, FINANCIALS)

    assert outcome.success
    assert outcome.result.result == expected


@pytest.mark.parametrize("question", [
    "What is the average salary of the engineers?",
    "What is the percentage increase in profit from 2022 to 2023?",
    "What is the total headcount in 2022 and 2023?",
])
def test_a_missing_operand_is_a_refusal_not_a_guess(question) -> None:
    outcome = calculation.compute(question, FINANCIALS)

    assert not outcome.success
    assert "couldn't find" in outcome.message
    assert outcome.inputs is None


# ---------------------------------------------------------------------------
# EXTRACTION: nothing the passages don't state
# ---------------------------------------------------------------------------

PASSAGES = [
    {"content": "ZigBee operates in the 2.4 GHz frequency band and uses a mesh topology with data rates up to 250 kbps."},
    {"content": "Bluetooth Low Energy (BLE) operates at 2.4 GHz with a data rate of 1 Mbps and a range of about 50 metres."},
]


def _raw(*items) -> str:
    return json.dumps({"items": list(items)})


def _item(entity, **fields) -> dict:
    return {"entity": entity, "fields": {name: {"value": value, "source_id": source} for name, (value, source) in fields.items()}}


def test_an_entity_the_passages_never_mention_is_dropped_even_with_a_cited_value() -> None:
    raw = _raw(
        _item("ZigBee", frequency=("2.4 GHz", 1)),
        _item("United States", population=("837905265", 2), capital=("Washington, D.C.", 2)),  # what the live model invented
    )

    items = parse_extraction_response(raw, ["frequency"], 2, PASSAGES)

    assert [item.entity for item in items] == ["ZigBee"]


def test_a_number_that_is_not_in_the_cited_passage_becomes_not_stated() -> None:
    raw = _raw(_item("ZigBee", frequency=("2.4 GHz", 1), rate=("500 kbps", 1)))

    (item,) = parse_extraction_response(raw, ["frequency", "rate"], 2, PASSAGES)

    assert item.fields["frequency"].value == "2.4 GHz" and item.fields["frequency"].source_id == 1
    assert item.fields["rate"].value == "Not stated" and item.fields["rate"].source_id is None


def test_a_value_pinned_to_the_wrong_passage_becomes_not_stated() -> None:
    raw = _raw(_item("ZigBee", topology=("mesh topology", 1), rate=("1 Mbps", 1)))  # 1 Mbps is BLE's rate, in passage 2

    (item,) = parse_extraction_response(raw, ["topology", "rate"], 2, PASSAGES)

    assert item.fields["rate"].value == "Not stated"


def test_an_item_whose_every_field_is_unsupported_is_dropped() -> None:
    assert parse_extraction_response(_raw(_item("ZigBee", rate=("999 kbps", 1))), ["rate"], 2, PASSAGES) == []
    assert parse_extraction_response(_raw(_item("ZigBee", rate=(None, None))), ["rate"], 2, PASSAGES) == []


def test_formatting_differences_do_not_reject_a_correct_value() -> None:
    passages = [{"content": "Acme revenue reached 1,250,000 dollars. The Acme battery is powered by two AA batteries."}]
    raw = _raw(_item("Acme", revenue=("1250000", 1), power=("Powered by AA batteries", 1)))

    (item,) = parse_extraction_response(raw, ["revenue", "power"], 1, passages)

    assert item.fields["revenue"].value == "1250000" and item.fields["power"].source_id == 1


def test_extraction_through_chat_refuses_the_live_hallucination(library) -> None:
    fabricated = _raw(
        _item("United States", population=("837905265", 2), capital=("Washington, D.C.", 2)),
        _item("Canada", population=("38000000", 2), capital=("Ottawa", 2)),
    )
    result, _ = _ask(library, "Extract the population and capital city for each country", llm=fabricated)

    assert result["answer"] == "I couldn't extract structured data for that request from the selected document."
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# Exam topic: a topic the document never mentions is not silently swapped
# ---------------------------------------------------------------------------

LONG_PASSAGE = "MQTT is a lightweight publish subscribe messaging protocol. Sensors publish readings to topics on a broker and gateways subscribe to them. " * 2


class _NearestNeighbours:
    """Real vector search always returns its nearest chunks -- including for a
    topic the document never mentions. That is the case the topic gate exists for."""

    def __init__(self, ids) -> None:
        self.ids = ids

    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [self.ids[:n_results]]}


def _quiz_library(db, monkeypatch):
    monkeypatch.setattr(learning_module, "get_collection", lambda: _NearestNeighbours([f"mqtt-{page}-0" for page in range(1, 7)]))
    db.add(Document(id="mqtt", filename="mqtt.pdf", status="ready", page_count=6))
    db.add_all([Chunk(id=f"mqtt-{page}-0", document_id="mqtt", page_number=page, content=f"{LONG_PASSAGE} (page {page})", start_offset=0, end_offset=len(LONG_PASSAGE)) for page in range(1, 7)])
    db.commit()


def _quiz(db, monkeypatch, **request):
    calls = []

    async def fake_generate(messages, **kwargs):
        calls.append(messages)
        mcq = {"type": "mcq", "question": "What does an MQTT broker do?", "options": ["Routes messages", "Renders pages", "Compiles code", "Stores images"], "answer": "Routes messages", "explanation": "It routes messages.", "source_ids": [1]}
        return json.dumps({"questions": [mcq]})

    monkeypatch.setattr("app.services.exam.generate", fake_generate)
    with use_providers(embedding=FakeEmbeddingProvider()):
        result = asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["mqtt"], question_type="mcq", count=1, **request), db=db))
    return result, calls


def test_a_quiz_on_a_topic_the_document_never_mentions_is_refused_without_calling_the_model(db_session, monkeypatch) -> None:
    _quiz_library(db_session, monkeypatch)

    with pytest.raises(HTTPException) as error:
        _quiz(db_session, monkeypatch, topic="quantum computing")

    assert error.value.status_code == 400
    assert 'doesn\'t appear to cover "quantum computing"' in error.value.detail


def test_a_quiz_on_a_topic_the_document_covers_is_generated(db_session, monkeypatch) -> None:
    _quiz_library(db_session, monkeypatch)

    result, calls = _quiz(db_session, monkeypatch, topic="MQTT brokers")  # 'brokers' ~ 'broker'

    assert result["generated"] == 1 and calls


def test_go_freely_may_quiz_beyond_the_document_and_a_selection_needs_no_topic_match(db_session, monkeypatch) -> None:
    _quiz_library(db_session, monkeypatch)

    result, _ = _quiz(db_session, monkeypatch, topic="quantum computing", grounding_mode="free")
    assert result["generated"] == 1
    selected = _quiz(db_session, monkeypatch, topic="quantum computing", selected_text=LONG_PASSAGE, selection_page=1)[0]
    assert selected["generated"] == 1


# ---------------------------------------------------------------------------
# Templates written for one kind of document
# ---------------------------------------------------------------------------


def _template(db, template_id, **extra):
    return _ask(db, "Apply the template", llm=json.dumps({"answer": "## Facts\nInvented legal facts. [1]", "source_ids": [1]}), mode="template", template_id=template_id, **extra)


def test_a_case_brief_of_a_document_that_is_not_a_legal_case_is_refused_not_invented(library) -> None:
    result, model = _template(library, "builtin:case-brief")

    assert not model.calls  # no model call to fabricate a "rule of law"
    assert "can't be applied" in result["answer"] and "legal case" in result["answer"]
    assert result["citations"] == []
    assert result["template"]["id"] == "builtin:case-brief"


def test_a_case_brief_of_an_actual_legal_document_is_generated(library) -> None:
    text = "The plaintiff sued the defendant in the district court; after trial the judge ruled for the plaintiff."
    library.add(Document(id="case", filename="case.pdf", status="ready", page_count=1))
    library.add(Chunk(id="case-1-0", document_id="case", page_number=1, content=text, start_offset=0, end_offset=len(text)))
    library.commit()

    result, model = _template(library, "builtin:case-brief", document_ids=("case",))

    assert model.calls
    assert result["citations"][0]["document_id"] == "case"


def test_templates_without_a_fit_rule_apply_to_any_document(library) -> None:
    for template_id in ("builtin:executive-summary", "builtin:study-guide", "builtin:lecture-notes", "builtin:research-paper-review"):
        result, model = _template(library, template_id)
        assert model.calls, template_id
        assert "can't be applied" not in result["answer"]


def test_a_custom_template_is_never_refused_for_fit(library) -> None:
    custom = templates.create_template(library, "Risk Register", "Risks", "List every risk the document mentions.", "## Risks")

    result, model = _template(library, custom.id)

    assert model.calls and "can't be applied" not in result["answer"]


# ---------------------------------------------------------------------------
# the shared term-matching used above
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terms, text, covered", [
    (["sensor"], "temperature sensors and humidity sensors", True),  # plural
    (["computing"], "a personal computer", True),
    (["quantum"], "the quantity of gateways", False),  # look-alike, different word
    (["quantum", "computing"], "computing at the edge", False),  # two-word topics need both
    (["mqtt", "quality", "service"], "the quality of service levels of MQTT", True),  # 3 of 3
    (["mqtt", "latency", "throughput"], "MQTT latency", True),  # 2 of 3 = 60%
    (["mqtt", "latency", "throughput", "jitter"], "MQTT latency", False),  # 2 of 4
    ([], "anything", True),
])
def test_topic_term_matching(terms, text, covered) -> None:
    assert evidence.covers_terms(terms, text) is covered


def test_a_topic_of_only_filler_words_is_not_checked() -> None:
    assert evidence.topic_is_covered("the of and", ["unrelated text"])
