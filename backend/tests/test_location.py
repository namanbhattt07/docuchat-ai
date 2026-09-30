import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Chunk, Document
from app.services.retrieval import retrieve_location
from app.services.retrieval_types import RetrievalMode, RetrievalPlan

# Group 4 LOCATION mode: keyword-only lookup (see retrieve_location's own
# docstring for why vector search is deliberately skipped). No embedding
# mock is needed anywhere in this file -- that's the point.


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed_document(session, document_id="doc-1", filename="protocols.pdf"):
    session.add(Document(id=document_id, filename=filename, status="ready"))
    session.flush()


def _seed_chunk(session, document_id, chunk_id, page, content, section=None):
    session.add(Chunk(
        id=chunk_id, document_id=document_id, page_number=page, content=content,
        section=section, start_offset=0, end_offset=len(content),
    ))


def _plan(topic: str, document_ids=("doc-1",), top_k=6) -> RetrievalPlan:
    return RetrievalPlan(mode=RetrievalMode.LOCATION, query=topic, document_ids=list(document_ids) if document_ids else None, location_topic=topic, top_k=top_k)


def test_retrieve_location_finds_the_page_a_term_is_mentioned_on(db_session) -> None:
    _seed_document(db_session)
    _seed_chunk(db_session, "doc-1", "c1", 59, "ZigBee operates in the 2.4 GHz frequency band.", section="Wireless Communication Protocols")
    _seed_chunk(db_session, "doc-1", "c2", 12, "MQTT is a lightweight publish/subscribe protocol.", section="Application Layer Protocols")
    db_session.commit()

    results = retrieve_location(db_session, _plan("ZigBee"))

    assert len(results) == 1
    assert results[0]["page_number"] == 59
    assert results[0]["section"] == "Wireless Communication Protocols"


def test_retrieve_location_is_case_insensitive(db_session) -> None:
    _seed_document(db_session)
    _seed_chunk(db_session, "doc-1", "c1", 5, "The MQTT protocol is widely used in IoT.")
    db_session.commit()

    results = retrieve_location(db_session, _plan("mqtt"))

    assert len(results) == 1
    assert results[0]["page_number"] == 5


def test_retrieve_location_returns_empty_for_a_term_never_mentioned(db_session) -> None:
    _seed_document(db_session)
    _seed_chunk(db_session, "doc-1", "c1", 1, "This document only discusses sensors and actuators.")
    db_session.commit()

    assert retrieve_location(db_session, _plan("quantum entanglement")) == []


def test_retrieve_location_returns_empty_when_no_documents_exist(db_session) -> None:
    assert retrieve_location(db_session, _plan("anything", document_ids=None)) == []


def test_retrieve_location_finds_multiple_distinct_pages(db_session) -> None:
    _seed_document(db_session)
    _seed_chunk(db_session, "doc-1", "c1", 59, "ZigBee operates in the 2.4 GHz band.", section="Wireless Communication Protocols")
    _seed_chunk(db_session, "doc-1", "c2", 61, "The ZigBee architecture includes coordinators, routers and end devices.", section="ZigBee Architecture")
    db_session.commit()

    results = retrieve_location(db_session, _plan("ZigBee"))

    pages = {item["page_number"] for item in results}
    assert pages == {59, 61}


def test_retrieve_location_respects_top_k(db_session) -> None:
    _seed_document(db_session)
    for page in range(1, 10):
        _seed_chunk(db_session, "doc-1", f"c{page}", page, f"Page {page} mentions ZigBee explicitly here.")
    db_session.commit()

    results = retrieve_location(db_session, _plan("ZigBee", top_k=3))

    assert len(results) == 3


def test_retrieve_location_scopes_to_selected_document_ids(db_session) -> None:
    _seed_document(db_session, "doc-1")
    _seed_document(db_session, "doc-2", "other.pdf")
    _seed_chunk(db_session, "doc-1", "c1", 1, "ZigBee is discussed here.")
    _seed_chunk(db_session, "doc-2", "c2", 1, "ZigBee is also discussed in this other document.")
    db_session.commit()

    results = retrieve_location(db_session, _plan("ZigBee", document_ids=["doc-1"]))

    assert {item["document_id"] for item in results} == {"doc-1"}
