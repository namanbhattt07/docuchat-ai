import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import PageText
from app.services.citations import locate_passage_bbox


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed_page(session, document_id: str, page_number: int, content: str, blocks: list[dict]) -> None:
    session.add(PageText(
        id=f"{document_id}-{page_number}",
        document_id=document_id,
        page_number=page_number,
        content=content,
        blocks_json=json.dumps(blocks),
    ))
    session.commit()


def test_locate_passage_bbox_exact_substring_match(db_session) -> None:
    content = "Overfitting happens when a model memorizes noise instead of signal."
    start = content.index("model memorizes noise")
    end = start + len("model memorizes noise")
    _seed_page(db_session, "doc-1", 3, content, [{"start": start, "end": end, "bbox": [1.0, 2.0, 3.0, 4.0]}])

    bbox = locate_passage_bbox(db_session, "doc-1", 3, "model memorizes noise")

    assert bbox == [1.0, 2.0, 3.0, 4.0]


def test_locate_passage_bbox_is_case_insensitive(db_session) -> None:
    content = "Regularization Techniques Reduce Overfitting In Deep Networks."
    start = content.lower().index("reduce overfitting")
    end = start + len("reduce overfitting")
    _seed_page(db_session, "doc-1", 1, content, [{"start": start, "end": end, "bbox": [5.0, 5.0, 50.0, 15.0]}])

    bbox = locate_passage_bbox(db_session, "doc-1", 1, "reduce overfitting")

    assert bbox == [5.0, 5.0, 50.0, 15.0]


def test_locate_passage_bbox_falls_back_to_fuzzy_match_on_minor_differences(db_session) -> None:
    content = "The quick brown fox jumps over the lazy dog near the riverbank."
    start = content.index("quick brown fox jumps over the lazy dog")
    end = start + len("quick brown fox jumps over the lazy dog")
    _seed_page(db_session, "doc-1", 2, content, [{"start": start, "end": end, "bbox": [0.0, 0.0, 10.0, 10.0]}])

    # Slightly reworded/mangled compared to the stored page text (extra
    # whitespace collapse + a swapped word), simulating a chunk whose
    # original line breaks or spacing don't match the page text verbatim.
    bbox = locate_passage_bbox(db_session, "doc-1", 2, "quick brown fox   jumps over the lazy dog")

    assert bbox == [0.0, 0.0, 10.0, 10.0]


def test_locate_passage_bbox_returns_none_when_page_missing(db_session) -> None:
    assert locate_passage_bbox(db_session, "missing-doc", 1, "anything") is None


def test_locate_passage_bbox_returns_none_for_unrelated_text(db_session) -> None:
    _seed_page(db_session, "doc-1", 1, "This page is entirely about cooking recipes.", [])

    bbox = locate_passage_bbox(db_session, "doc-1", 1, "quantum entanglement in superconductors")

    assert bbox is None


def test_locate_passage_bbox_returns_none_for_blank_text(db_session) -> None:
    _seed_page(db_session, "doc-1", 1, "Some content.", [])

    assert locate_passage_bbox(db_session, "doc-1", 1, "   ") is None
