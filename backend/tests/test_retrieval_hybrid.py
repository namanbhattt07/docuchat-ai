import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Chunk, Document
from app.services import retrieval as retrieval_module
from app.services.retrieval import (
    deduplicate,
    fuse_scores,
    list_known_sections,
    retrieve_fact,
    retrieve_multi_document,
    retrieve_page,
    retrieve_section,
    retrieve_selection,
    retrieve_with_plan,
    select_diverse,
)
from app.services.retrieval_types import RetrievalMode, RetrievalPlan


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


class FakeCollection:
    """Vector store stand-in: returns whatever chunk_id ordering the test
    configures, so retrieval tests never depend on a real embedding model.
    """

    def __init__(self, ordered_ids: list[str]) -> None:
        self.ordered_ids = ordered_ids

    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [self.ordered_ids[:n_results]]}


def _seed_chunk(session, document_id, chunk_id, page, content, section=None, start_offset=0, end_offset=None):
    session.add(Chunk(
        id=chunk_id, document_id=document_id, page_number=page, content=content, section=section,
        start_offset=start_offset, end_offset=end_offset if end_offset is not None else len(content),
    ))


def _seed_document(session, document_id="doc-1", filename="test.pdf"):
    document = Document(id=document_id, filename=filename, status="ready")
    session.add(document)
    session.flush()
    return document


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    async def _embed(texts):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(retrieval_module, "embed", _embed)


# ---------- fusion ----------

def test_fuse_scores_rewards_items_ranked_highly_in_both_lists() -> None:
    fused = fuse_scores(["a", "b", "c"], ["a", "c", "d"], vector_weight=1.0, keyword_weight=1.0, rrf_k=60)
    assert fused["a"] > fused["b"]  # a is #1 in both lists
    assert fused["a"] > fused["c"]  # c is only #2 vector + #2 keyword vs a's #1 + #1


def test_fuse_scores_includes_keyword_only_hits() -> None:
    fused = fuse_scores(["a"], ["z"], vector_weight=1.0, keyword_weight=1.0, rrf_k=60)
    assert "z" in fused


def test_fuse_scores_weights_are_configurable() -> None:
    heavy_vector = fuse_scores(["a"], ["b"], vector_weight=10.0, keyword_weight=1.0, rrf_k=60)
    heavy_keyword = fuse_scores(["a"], ["b"], vector_weight=1.0, keyword_weight=10.0, rrf_k=60)
    assert heavy_vector["a"] > heavy_vector["b"]
    assert heavy_keyword["b"] > heavy_keyword["a"]


# ---------- dedup ----------

def _source(chunk_id, document_id, page, content, score, start=0, end=None):
    return {
        "chunk_id": chunk_id, "document_id": document_id, "page_number": page, "content": content,
        "score": score, "start_offset": start, "end_offset": end if end is not None else len(content),
        "filename": "test.pdf", "section": None, "bbox": None, "source_type": "text",
    }


def test_deduplicate_collapses_heavily_overlapping_chunks_on_same_page() -> None:
    a = _source("a", "doc-1", 1, "x" * 100, score=5.0, start=0, end=100)
    b = _source("b", "doc-1", 1, "y" * 90, score=3.0, start=10, end=100)  # 90/90 = 100% overlap of the shorter span

    kept = deduplicate([a, b], overlap_ratio=0.6)

    assert len(kept) == 1
    assert kept[0]["chunk_id"] == "a"  # higher score wins


def test_deduplicate_keeps_distinct_chunks_from_the_same_page() -> None:
    a = _source("a", "doc-1", 1, "alpha content", score=5.0, start=0, end=200)
    b = _source("b", "doc-1", 1, "beta content", score=3.0, start=700, end=900)

    kept = deduplicate([a, b], overlap_ratio=0.6)

    assert {item["chunk_id"] for item in kept} == {"a", "b"}


def test_deduplicate_keeps_chunks_from_different_pages_even_if_identical() -> None:
    a = _source("a", "doc-1", 1, "same text", score=5.0)
    b = _source("b", "doc-1", 2, "same text", score=3.0)

    kept = deduplicate([a, b], overlap_ratio=0.6)

    assert {item["chunk_id"] for item in kept} == {"a", "b"}


# ---------- diversity ----------

def test_select_diverse_spreads_across_pages_instead_of_one_dominant_page() -> None:
    sources = [_source(f"p1-{i}", "doc-1", 1, "x", score=10 - i) for i in range(5)]
    sources += [_source("p2-0", "doc-1", 2, "x", score=4.5)]
    sources += [_source("p3-0", "doc-1", 3, "x", score=4.0)]

    selected = select_diverse(sources, limit=3)

    pages = {item["page_number"] for item in selected}
    assert pages == {1, 2, 3}


def test_select_diverse_falls_back_to_plain_top_n_when_not_over_limit() -> None:
    sources = [_source("a", "doc-1", 1, "x", score=5.0), _source("b", "doc-1", 2, "x", score=3.0)]

    selected = select_diverse(sources, limit=5)

    assert [item["chunk_id"] for item in selected] == ["a", "b"]


def test_select_diverse_supports_custom_group_key_for_multi_document() -> None:
    sources = [_source(f"a{i}", "doc-a", 1, "x", score=10 - i) for i in range(4)]
    sources += [_source("b0", "doc-b", 1, "x", score=1.0)]

    selected = select_diverse(sources, limit=2, group_key=lambda item: item["document_id"])

    assert {item["document_id"] for item in selected} == {"doc-a", "doc-b"}


# ---------- mode-specific retrieval ----------

def test_retrieve_page_prioritizes_requested_page(db_session) -> None:
    document = _seed_document(db_session)
    _seed_chunk(db_session, document.id, "c-p1", 1, "Page one content about IoT.")
    _seed_chunk(db_session, document.id, "c-p2a", 2, "Page two content about sensors.")
    _seed_chunk(db_session, document.id, "c-p2b", 2, "Page two content about MQTT.")
    db_session.commit()

    sources = retrieve_page(db_session, RetrievalPlan(mode=RetrievalMode.PAGE, query="", document_ids=[document.id], page=2, top_k=8, diversity=False))

    assert [item["page_number"] for item in sources] == [2, 2]


def test_retrieve_page_pads_with_neighbors_when_page_is_sparse(db_session) -> None:
    document = _seed_document(db_session)
    _seed_chunk(db_session, document.id, "c-p1", 1, "Page one content.")
    _seed_chunk(db_session, document.id, "c-p2", 2, "Page two content, the only chunk on this page.")
    _seed_chunk(db_session, document.id, "c-p3", 3, "Page three content.")
    db_session.commit()

    sources = retrieve_page(db_session, RetrievalPlan(mode=RetrievalMode.PAGE, query="", document_ids=[document.id], page=2, top_k=8, diversity=False))

    pages = {item["page_number"] for item in sources}
    assert 2 in pages
    assert pages.issubset({1, 2, 3})
    assert len(sources) > 1


def test_retrieve_section_matches_chunks_by_heading(db_session) -> None:
    document = _seed_document(db_session)
    _seed_chunk(db_session, document.id, "c-sensors", 3, "Sensors detect physical phenomena.", section="Sensors")
    _seed_chunk(db_session, document.id, "c-mqtt", 5, "MQTT is a lightweight protocol.", section="MQTT Protocol")
    db_session.commit()

    plan = RetrievalPlan(mode=RetrievalMode.SECTION, query="what does it say", document_ids=[document.id], section="Sensors", top_k=8, diversity=False)
    sources = asyncio.run(retrieve_section(db_session, FakeCollection([]), plan))

    assert len(sources) == 1
    assert sources[0]["chunk_id"] == "c-sensors"


def test_retrieve_section_falls_back_to_hybrid_search_when_no_heading_matches(db_session) -> None:
    document = _seed_document(db_session)
    _seed_chunk(db_session, document.id, "c-1", 1, "This chunk talks about the Sensors used in IoT deployments.")
    db_session.commit()

    plan = RetrievalPlan(mode=RetrievalMode.SECTION, query="what does it say", document_ids=[document.id], section="Sensors", top_k=8, diversity=False)
    sources = asyncio.run(retrieve_section(db_session, FakeCollection(["c-1"]), plan))

    assert len(sources) == 1  # never an empty context just because the section name wasn't a literal heading


def test_retrieve_selection_puts_selected_text_first(db_session) -> None:
    document = _seed_document(db_session)
    _seed_chunk(db_session, document.id, "c-1", 1, "Supporting context chunk about MQTT.")
    db_session.commit()

    plan = RetrievalPlan(
        mode=RetrievalMode.SELECTION, query="", document_ids=[document.id],
        selected_text="MQTT is a lightweight publish/subscribe protocol.", page=1, top_k=4, diversity=False,
    )
    sources = asyncio.run(retrieve_selection(db_session, FakeCollection(["c-1"]), plan))

    assert sources[0]["source_type"] == "selection"
    assert sources[0]["content"] == "MQTT is a lightweight publish/subscribe protocol."
    assert any(item["chunk_id"] == "c-1" for item in sources[1:])


def test_retrieve_multi_document_spreads_across_documents(db_session) -> None:
    doc_a = _seed_document(db_session, "doc-a", "a.pdf")
    doc_b = _seed_document(db_session, "doc-b", "b.pdf")
    for i in range(5):
        _seed_chunk(db_session, doc_a.id, f"a-{i}", 1, f"Document A chunk {i} about IoT.")
    _seed_chunk(db_session, doc_b.id, "b-0", 1, "Document B chunk about IoT.")
    db_session.commit()

    plan = RetrievalPlan(mode=RetrievalMode.MULTI_DOCUMENT, query="IoT", document_ids=[doc_a.id, doc_b.id], top_k=2, diversity=True)
    collection = FakeCollection([f"a-{i}" for i in range(5)] + ["b-0"])
    sources = asyncio.run(retrieve_multi_document(db_session, collection, plan))

    assert {item["document_id"] for item in sources} == {"doc-a", "doc-b"}


def test_retrieve_fact_returns_empty_for_unknown_document(db_session) -> None:
    plan = RetrievalPlan(mode=RetrievalMode.FACT, query="anything", document_ids=["missing"], top_k=8, diversity=True)
    sources = asyncio.run(retrieve_fact(db_session, FakeCollection([]), plan))
    assert sources == []


def test_retrieve_fact_fuses_vector_and_keyword_hits(db_session) -> None:
    document = _seed_document(db_session)
    _seed_chunk(db_session, document.id, "c-mqtt", 1, "MQTT protocol overview for IoT devices.")
    _seed_chunk(db_session, document.id, "c-unrelated", 2, "This is about something else entirely, cats.")
    db_session.commit()

    plan = RetrievalPlan(mode=RetrievalMode.FACT, query="MQTT protocol", document_ids=[document.id], top_k=8, diversity=False)
    sources = asyncio.run(retrieve_fact(db_session, FakeCollection(["c-unrelated", "c-mqtt"]), plan))

    # Keyword relevance should be able to pull the exact-term chunk to the
    # top even though the fake vector store ranked it second.
    assert sources[0]["chunk_id"] == "c-mqtt"


def test_retrieve_with_plan_dispatches_to_overview(db_session) -> None:
    document = _seed_document(db_session)
    _seed_chunk(db_session, document.id, "c-1", 1, "Page one.")
    _seed_chunk(db_session, document.id, "c-2", 2, "Page two.")
    db_session.commit()

    plan = RetrievalPlan(mode=RetrievalMode.OVERVIEW, query="summarize this", document_ids=[document.id], top_k=2, diversity=True)
    sources = asyncio.run(retrieve_with_plan(db_session, FakeCollection([]), plan))

    assert {item["page_number"] for item in sources} == {1, 2}


def test_list_known_sections_returns_distinct_headings(db_session) -> None:
    document = _seed_document(db_session)
    _seed_chunk(db_session, document.id, "c-1", 1, "a", section="Sensors")
    _seed_chunk(db_session, document.id, "c-2", 2, "b", section="Sensors")
    _seed_chunk(db_session, document.id, "c-3", 3, "c", section="MQTT Protocol")
    _seed_chunk(db_session, document.id, "c-4", 4, "d", section=None)
    db_session.commit()

    sections = list_known_sections(db_session, [document.id])

    assert set(sections) == {"Sensors", "MQTT Protocol"}
