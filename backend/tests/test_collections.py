import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.api.v1.collections as collections_api
from app.db import Base
from app.models import CollectionDocument, Document
from app.services import collections as collections_service


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _add_document(db, document_id: str, filename: str = "doc.pdf") -> Document:
    document = Document(id=document_id, filename=filename, status="ready")
    db.add(document)
    db.commit()
    return document


# ---------------------------------------------------------------------------
# Service-layer CRUD
# ---------------------------------------------------------------------------


def test_create_collection_returns_empty_document_list(db_session) -> None:
    result = collections_service.create_collection(db_session, "IoT Notes")

    assert result["name"] == "IoT Notes"
    assert result["documents"] == []
    assert result["id"]


def test_rename_collection_updates_name(db_session) -> None:
    created = collections_service.create_collection(db_session, "IoT Notes")

    renamed = collections_service.rename_collection(db_session, created["id"], "IoT Notes v2")

    assert renamed["name"] == "IoT Notes v2"
    assert collections_service.get_collection(db_session, created["id"])["name"] == "IoT Notes v2"


def test_rename_missing_collection_raises_not_found(db_session) -> None:
    with pytest.raises(collections_service.CollectionNotFound):
        collections_service.rename_collection(db_session, "missing", "New name")


def test_add_document_to_collection_creates_membership(db_session) -> None:
    collection = collections_service.create_collection(db_session, "IoT Notes")
    _add_document(db_session, "doc-1", "Unit 1.pdf")

    result = collections_service.add_document(db_session, collection["id"], "doc-1")

    assert [doc["id"] for doc in result["documents"]] == ["doc-1"]
    assert collections_service.member_document_ids(db_session, collection["id"]) == {"doc-1"}


def test_add_document_twice_is_idempotent(db_session) -> None:
    collection = collections_service.create_collection(db_session, "IoT Notes")
    _add_document(db_session, "doc-1")

    collections_service.add_document(db_session, collection["id"], "doc-1")
    result = collections_service.add_document(db_session, collection["id"], "doc-1")

    assert len(result["documents"]) == 1
    memberships = db_session.scalars(select(CollectionDocument)).all()
    assert len(memberships) == 1


def test_add_document_to_missing_collection_raises(db_session) -> None:
    _add_document(db_session, "doc-1")
    with pytest.raises(collections_service.CollectionNotFound):
        collections_service.add_document(db_session, "missing", "doc-1")


def test_add_missing_document_raises(db_session) -> None:
    collection = collections_service.create_collection(db_session, "IoT Notes")
    with pytest.raises(collections_service.DocumentNotFound):
        collections_service.add_document(db_session, collection["id"], "missing-doc")


def test_remove_document_deletes_membership_but_not_document(db_session) -> None:
    collection = collections_service.create_collection(db_session, "IoT Notes")
    _add_document(db_session, "doc-1")
    collections_service.add_document(db_session, collection["id"], "doc-1")

    result = collections_service.remove_document(db_session, collection["id"], "doc-1")

    assert result["documents"] == []
    assert db_session.get(Document, "doc-1") is not None  # the underlying document survives


def test_remove_document_not_in_collection_is_a_noop(db_session) -> None:
    collection = collections_service.create_collection(db_session, "IoT Notes")
    _add_document(db_session, "doc-1")

    result = collections_service.remove_document(db_session, collection["id"], "doc-1")

    assert result["documents"] == []


def test_delete_collection_removes_membership_but_not_documents(db_session) -> None:
    collection = collections_service.create_collection(db_session, "IoT Notes")
    _add_document(db_session, "doc-1")
    _add_document(db_session, "doc-2")
    collections_service.add_document(db_session, collection["id"], "doc-1")
    collections_service.add_document(db_session, collection["id"], "doc-2")

    collections_service.delete_collection(db_session, collection["id"])

    assert db_session.get(Document, "doc-1") is not None
    assert db_session.get(Document, "doc-2") is not None
    assert db_session.scalars(select(CollectionDocument)).all() == []
    with pytest.raises(collections_service.CollectionNotFound):
        collections_service.get_collection(db_session, collection["id"])


def test_a_document_can_belong_to_multiple_collections(db_session) -> None:
    iot = collections_service.create_collection(db_session, "IoT Notes")
    ml = collections_service.create_collection(db_session, "ML Notes")
    _add_document(db_session, "shared-doc")

    collections_service.add_document(db_session, iot["id"], "shared-doc")
    collections_service.add_document(db_session, ml["id"], "shared-doc")

    assert collections_service.member_document_ids(db_session, iot["id"]) == {"shared-doc"}
    assert collections_service.member_document_ids(db_session, ml["id"]) == {"shared-doc"}


def test_list_collections_returns_every_collection_with_members(db_session) -> None:
    iot = collections_service.create_collection(db_session, "IoT Notes")
    collections_service.create_collection(db_session, "ML Notes")
    _add_document(db_session, "doc-1")
    collections_service.add_document(db_session, iot["id"], "doc-1")

    listed = collections_service.list_collections(db_session)

    names = {item["name"]: item["documents"] for item in listed}
    assert [doc["id"] for doc in names["IoT Notes"]] == ["doc-1"]
    assert names["ML Notes"] == []


# ---------------------------------------------------------------------------
# API router wiring (thin -- most logic already covered at the service layer)
# ---------------------------------------------------------------------------


def test_api_create_and_get_collection_round_trips(db_session) -> None:
    created = collections_api.create_collection(collections_api.CreateCollectionRequest(name="IoT Notes"), db=db_session)

    fetched = collections_api.get_collection(created["id"], db=db_session)

    assert fetched["name"] == "IoT Notes"


def test_api_get_missing_collection_raises_404(db_session) -> None:
    with pytest.raises(HTTPException) as exc_info:
        collections_api.get_collection("missing", db=db_session)
    assert exc_info.value.status_code == 404


def test_api_delete_missing_collection_raises_404(db_session) -> None:
    with pytest.raises(HTTPException) as exc_info:
        collections_api.delete_collection("missing", db=db_session)
    assert exc_info.value.status_code == 404


def test_api_add_document_to_missing_collection_raises_404(db_session) -> None:
    _add_document(db_session, "doc-1")
    with pytest.raises(HTTPException) as exc_info:
        collections_api.add_document("missing", collections_api.AddDocumentRequest(document_id="doc-1"), db=db_session)
    assert exc_info.value.status_code == 404
