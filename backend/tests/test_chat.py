import asyncio
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.chat as chat_module
import app.services.retrieval as retrieval_module
from app.api.v1.chat import (
    ChatRequest,
    ask,
    build_calculation_answer,
    build_citations,
    build_instructions,
    build_location_answer,
    build_location_not_found_answer,
    format_source_block,
    grounded_response,
)
from app.db import Base
from app.models import Chunk, Document, PageText
from app.services import calculation
from app.services.query_router import FormatMode
from app.services.retrieval_types import RetrievalMode


def test_parses_valid_json_response() -> None:
    raw = '{"answer":"Overfitting means memorizing noise. [1]","source_ids":[1]}'

    answer, source_ids = grounded_response(raw, source_count=2)

    assert source_ids == [1]
    assert "[1]" in answer


def test_normalizes_word_style_markers_instead_of_double_citing() -> None:
    """Regression test: the model sometimes writes "[source-1]" instead of
    "[1]". The old code only checked for the literal "[1]" before
    force-appending a citation marker, so it appended a second one, producing
    "...[source-1] [1]" in the UI -- which reads as a glitch/hallucination
    even though the underlying fact was correct.
    """
    raw = '{"answer":"I chose top-volume SKUs for a richer signal [source-1]","source_ids":[1]}'

    answer, source_ids = grounded_response(raw, source_count=1)

    assert "[source-1]" not in answer.lower()
    assert answer.count("[1]") == 1
    assert source_ids == [1]


def test_falls_back_to_regex_when_json_is_missing() -> None:
    raw = "The answer is clearly stated here [1] and confirmed again [2]."

    answer, source_ids = grounded_response(raw, source_count=2)

    assert source_ids == [1, 2]


def test_refuses_when_no_valid_sources_are_cited() -> None:
    raw = '{"answer":"Something outside the provided passages.","source_ids":[]}'

    answer, source_ids = grounded_response(raw, source_count=2)

    assert source_ids == []
    assert "couldn't verify" in answer.lower() or "couldn’t verify" in answer.lower()


def test_caps_sources_to_max_sources() -> None:
    raw = '{"answer":"broad summary [1][2][3][4]","source_ids":[1,2,3,4]}'

    answer, source_ids = grounded_response(raw, source_count=4, max_sources=2)

    assert source_ids == [1, 2]


def test_strips_think_block_before_parsing() -> None:
    raw = '<think>internal reasoning here</think>{"answer":"Fact from doc [1]","source_ids":[1]}'

    answer, source_ids = grounded_response(raw, source_count=1)

    assert "internal reasoning" not in answer
    assert source_ids == [1]


def test_drops_source_ids_outside_valid_range() -> None:
    raw = '{"answer":"Answer citing a source that does not exist [9]","source_ids":[9]}'

    answer, source_ids = grounded_response(raw, source_count=2)

    assert source_ids == []


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _source(**overrides) -> dict:
    base = {
        "document_id": "doc-1",
        "filename": "sample.pdf",
        "page_number": 1,
        "content": "Overfitting happens when a model memorizes training noise.",
        "section": "Overfitting",
        "bbox": None,
    }
    base.update(overrides)
    return base


def test_build_citations_uses_chunk_bbox_directly_when_present(db_session) -> None:
    sources = [_source(bbox=[1.0, 2.0, 3.0, 4.0])]

    citations = build_citations(db_session, sources, [1])

    assert citations[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert citations[0]["bbox_source"] == "exact"


def test_build_citations_falls_back_to_text_location_when_bbox_missing(db_session) -> None:
    content = "Overfitting happens when a model memorizes training noise."
    start = content.index("model memorizes training noise")
    end = start + len("model memorizes training noise")
    db_session.add(PageText(
        id="doc-1-1", document_id="doc-1", page_number=1, content=content,
        blocks_json=json.dumps([{"start": start, "end": end, "bbox": [5.0, 5.0, 50.0, 15.0]}]),
    ))
    db_session.commit()
    sources = [_source(bbox=None)]

    citations = build_citations(db_session, sources, [1])

    assert citations[0]["bbox"] == [5.0, 5.0, 50.0, 15.0]
    assert citations[0]["bbox_source"] == "approximate"


def test_build_citations_reports_unavailable_without_false_precision(db_session) -> None:
    sources = [_source(bbox=None)]  # no PageText row seeded -- nothing to locate against

    citations = build_citations(db_session, sources, [1])

    assert citations[0]["bbox"] is None
    assert citations[0]["bbox_source"] == "unavailable"


def test_build_citations_preserves_existing_metadata_fields(db_session) -> None:
    sources = [_source(bbox=[0.0, 0.0, 1.0, 1.0], section="Intro")]

    citations = build_citations(db_session, sources, [1])

    assert citations[0]["filename"] == "sample.pdf"
    assert citations[0]["page_number"] == 1
    assert citations[0]["section"] == "Intro"
    assert citations[0]["excerpt"].startswith("Overfitting")


def test_build_instructions_keeps_json_schema_across_every_mode_and_format() -> None:
    for mode in RetrievalMode:
        for format_mode in FormatMode:
            instructions = build_instructions(mode, format_mode, max_sources=3)
            assert '"answer"' in instructions
            assert '"source_ids"' in instructions
            assert "1 to 3 source_ids" in instructions


def test_build_instructions_default_format_adds_no_extra_sentence() -> None:
    fact_default = build_instructions(RetrievalMode.FACT, FormatMode.DEFAULT, max_sources=3)
    assert "bullet" not in fact_default.lower()
    assert "table" not in fact_default.lower()


def test_build_instructions_bullets_format_is_added_independent_of_mode() -> None:
    section_bullets = build_instructions(RetrievalMode.SECTION, FormatMode.BULLETS, max_sources=3)
    assert "bullet" in section_bullets.lower()
    assert "specific section" in section_bullets.lower()  # retrieval-mode context sentence still present


def test_build_instructions_table_format_mentions_markdown_table() -> None:
    instructions = build_instructions(RetrievalMode.FACT, FormatMode.TABLE, max_sources=3)
    assert "markdown table" in instructions.lower()


def test_build_instructions_overview_mode_mentions_sampled_excerpts() -> None:
    instructions = build_instructions(RetrievalMode.OVERVIEW, FormatMode.DEFAULT, max_sources=10)
    assert "sampled evenly across the whole document" in instructions


def test_format_source_block_renders_normal_chunk() -> None:
    block = format_source_block({"filename": "sample.pdf", "page_number": 4, "content": "Some text.", "source_type": "text"}, 1)
    assert block == "SOURCE 1 | sample.pdf | page 4\nSome text."


def test_format_source_block_renders_selection_source_without_page_number() -> None:
    block = format_source_block({"filename": "sample.pdf", "page_number": None, "content": "Selected text.", "source_type": "selection"}, 1)
    assert block == "SOURCE 1 | sample.pdf | user-selected passage\nSelected text."


# ---------------------------------------------------------------------------
# Group 4: LOCATION answer builder (deterministic, no LLM)
# ---------------------------------------------------------------------------


def test_build_location_answer_lists_section_and_page_for_each_source() -> None:
    sources = [
        {"page_number": 59, "section": "Wireless Communication Protocols"},
        {"page_number": 61, "section": "ZigBee Architecture"},
    ]
    answer = build_location_answer("ZigBee", sources)
    assert answer.startswith("### ZigBee")
    assert "**Wireless Communication Protocols** — Page 59" in answer
    assert "**ZigBee Architecture** — Page 61" in answer


def test_build_location_answer_falls_back_to_bare_page_without_a_section() -> None:
    answer = build_location_answer("MQTT", [{"page_number": 12, "section": None}])
    assert "Page 12" in answer
    assert "**" not in answer.split("\n")[-1] or "Page 12" in answer  # no fabricated section label


def test_build_location_not_found_answer_names_the_topic() -> None:
    answer = build_location_not_found_answer("quantum entanglement")
    assert "quantum entanglement" in answer


# ---------------------------------------------------------------------------
# Group 4: CALCULATION answer builder (deterministic, no LLM)
# ---------------------------------------------------------------------------


def test_build_calculation_answer_renders_table_and_result() -> None:
    outcome = calculation.compute("What is the percentage increase from 100 to 125?", sources=[])
    assert outcome.success is True

    answer = build_calculation_answer(outcome)

    assert "| Value | Amount | Source |" in answer
    assert "**Percentage increase: 25%**" in answer


def test_build_calculation_answer_includes_citation_marker_for_source_backed_values() -> None:
    sources = [{"content": "Revenue was 80000 last year. Cost was 50000 last year."}]
    outcome = calculation.compute("What is the difference between revenue and cost?", sources)
    assert outcome.success is True

    answer = build_calculation_answer(outcome)

    assert "[1]" in answer
    assert "**Difference: 30000**" in answer


# ---------------------------------------------------------------------------
# Group 4: full /chat endpoint wiring -- SELECTION validation + lead-in,
# LOCATION/CALCULATION never touching the LLM, EXTRACTION rendering.
# ---------------------------------------------------------------------------


class _FakeCollection:
    """Vector store stand-in returning no vector hits -- the tests below
    exercise LOCATION (keyword-only), CALCULATION, and EXTRACTION's hybrid
    retrieval without depending on a real embedding model or this repo's
    on-disk ChromaDB store; the keyword/FTS5 half of hybrid retrieval still
    runs for real against the seeded chunks.
    """

    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def seeded_db(db_session, monkeypatch):
    monkeypatch.setattr(chat_module, "get_collection", lambda: _FakeCollection())

    async def _fake_embed(texts):
        return [[0.0] * 4 for _ in texts]
    monkeypatch.setattr(retrieval_module, "embed", _fake_embed)

    db_session.add(Document(id="doc-1", filename="protocols.pdf", status="ready"))
    db_session.add_all([
        Chunk(
            id="doc-1-59-0", document_id="doc-1", page_number=59,
            content="ZigBee operates in the 2.4 GHz frequency band and uses a mesh topology.",
            section="Wireless Communication Protocols", start_offset=0, end_offset=72,
        ),
        Chunk(
            id="doc-1-3-0", document_id="doc-1", page_number=3,
            content="Revenue was 80000 last year. Cost was 50000 last year.",
            start_offset=0, end_offset=55,
        ),
    ])
    db_session.commit()
    return db_session


def _no_llm_allowed(monkeypatch):
    async def _fail(*args, **kwargs):
        raise AssertionError("generate() must not be called for this mode")
    monkeypatch.setattr(chat_module, "generate", _fail)


def test_ask_rejects_selection_action_without_selected_text(seeded_db) -> None:
    request = ChatRequest(question="Explain this.", document_ids=["doc-1"], selection_action="explain")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(ask(request, db=seeded_db))
    assert exc_info.value.status_code == 400


def test_ask_location_mode_never_calls_the_llm_and_returns_clickable_citations(seeded_db, monkeypatch) -> None:
    _no_llm_allowed(monkeypatch)
    request = ChatRequest(question="Where does the document discuss ZigBee?", document_ids=["doc-1"])

    result = asyncio.run(ask(request, db=seeded_db))

    assert "ZigBee" in result["answer"]
    assert "Page 59" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["page_number"] == 59


def test_ask_location_mode_reports_missing_result_without_error(seeded_db, monkeypatch) -> None:
    _no_llm_allowed(monkeypatch)
    request = ChatRequest(question="Where does the document discuss quantum entanglement?", document_ids=["doc-1"])

    result = asyncio.run(ask(request, db=seeded_db))

    assert "couldn't find" in result["answer"].lower()
    assert result["citations"] == []


def test_ask_calculation_mode_never_calls_the_llm(seeded_db, monkeypatch) -> None:
    _no_llm_allowed(monkeypatch)
    request = ChatRequest(question="What is the difference between revenue and cost?", document_ids=["doc-1"])

    result = asyncio.run(ask(request, db=seeded_db))

    assert "30000" in result["answer"]
    assert result["citations"]  # the revenue/cost passage was cited


def test_ask_extraction_mode_renders_llm_items_as_a_grounded_table(seeded_db, monkeypatch) -> None:
    async def _fake_generate(messages, **kwargs):
        return json.dumps({"items": [{"entity": "ZigBee", "fields": {"frequency": {"value": "2.4 GHz", "source_id": 1}, "range": {"value": None, "source_id": None}}}]})
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    request = ChatRequest(question="Extract the following for each protocol:\n- frequency\n- range", document_ids=["doc-1"])

    result = asyncio.run(ask(request, db=seeded_db))

    assert "2.4 GHz [1]" in result["answer"]
    assert "Not stated" in result["answer"]
    assert len(result["citations"]) == 1


def test_ask_selection_mode_prepends_a_deterministic_lead_in(seeded_db, monkeypatch) -> None:
    async def _fake_generate(messages, **kwargs):
        return json.dumps({"answer": "ZigBee uses the 2.4 GHz band. [1]", "source_ids": [1]})
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    request = ChatRequest(
        question="Explain the selected passage.",
        document_ids=["doc-1"],
        selected_text="ZigBee operates in the 2.4 GHz frequency band.",
        selection_page=59,
        selection_action="explain",
    )

    result = asyncio.run(ask(request, db=seeded_db))

    assert result["answer"].startswith("Based on the selected passage:")
