import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Chunk, Document
from app.services import keyword_index


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def test_is_available_for_sqlite(db_session) -> None:
    assert keyword_index.is_available(db_session) is True


def test_search_finds_exact_term(db_session) -> None:
    keyword_index.index_chunks(
        db_session, "doc-1",
        ["doc-1-1-0", "doc-1-1-1"],
        ["The MQTT protocol is lightweight and used in IoT.", "This chunk is about cats and dogs only."],
        [1, 1],
    )
    db_session.commit()

    hits = keyword_index.search(db_session, ["mqtt", "protocol"], None, limit=10)

    assert hits
    assert hits[0][0] == "doc-1-1-0"


def test_search_ranks_phrase_match_above_partial_match(db_session) -> None:
    keyword_index.index_chunks(
        db_session, "doc-1",
        ["exact", "partial"],
        ["The MQTT protocol is the backbone of this system.", "MQTT is mentioned here, and protocol design elsewhere."],
        [1, 2],
    )
    db_session.commit()

    hits = keyword_index.search(db_session, ["mqtt", "protocol"], None, limit=10)
    ranked_ids = [chunk_id for chunk_id, _ in hits]

    assert ranked_ids[0] == "exact"


def test_search_scopes_to_document_ids(db_session) -> None:
    keyword_index.index_chunks(db_session, "doc-1", ["doc-1-a"], ["MQTT protocol overview."], [1])
    keyword_index.index_chunks(db_session, "doc-2", ["doc-2-a"], ["MQTT protocol overview."], [1])
    db_session.commit()

    hits = keyword_index.search(db_session, ["mqtt"], ["doc-1"], limit=10)

    assert [chunk_id for chunk_id, _ in hits] == ["doc-1-a"]


def test_search_returns_empty_for_empty_terms(db_session) -> None:
    assert keyword_index.search(db_session, [], None, limit=10) == []


def test_delete_document_removes_its_rows(db_session) -> None:
    keyword_index.index_chunks(db_session, "doc-1", ["doc-1-a"], ["MQTT protocol overview."], [1])
    db_session.commit()

    keyword_index.delete_document(db_session, "doc-1")
    db_session.commit()

    assert keyword_index.search(db_session, ["mqtt"], None, limit=10) == []


def test_ensure_indexed_backfills_chunks_created_before_fts_existed(db_session) -> None:
    document = Document(filename="test.pdf", status="ready")
    db_session.add(document)
    db_session.flush()
    db_session.add(Chunk(id="doc-1-1-0", document_id=document.id, page_number=1, content="MQTT protocol overview."))
    db_session.commit()  # no keyword_index.index_chunks call -- simulates a pre-Group-3 document

    assert keyword_index.search(db_session, ["mqtt"], [document.id], limit=10) == []

    keyword_index.ensure_indexed(db_session, document.id)

    hits = keyword_index.search(db_session, ["mqtt"], [document.id], limit=10)
    assert [chunk_id for chunk_id, _ in hits] == ["doc-1-1-0"]


def test_ensure_indexed_is_a_noop_when_already_indexed(db_session) -> None:
    document = Document(filename="test.pdf", status="ready")
    db_session.add(document)
    db_session.flush()
    db_session.add(Chunk(id="doc-1-1-0", document_id=document.id, page_number=1, content="MQTT protocol overview."))
    db_session.commit()
    keyword_index.index_chunks(db_session, document.id, ["doc-1-1-0"], ["MQTT protocol overview."], [1])
    db_session.commit()

    keyword_index.ensure_indexed(db_session, document.id)  # should not duplicate rows

    hits = keyword_index.search(db_session, ["mqtt"], [document.id], limit=10)
    assert len(hits) == 1
