from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Collection, CollectionDocument, Document

# Group 5: multi-document collections. A thin service layer over the
# Collection/CollectionDocument tables (see models.py) -- kept separate from
# the API router so both the HTTP layer and /chat's server-side scope
# validation (see api/v1/chat.py::_resolve_document_scope) can reuse the same
# membership logic instead of re-querying CollectionDocument by hand.


class CollectionNotFound(Exception):
    pass


class DocumentNotFound(Exception):
    pass


def _document_summary(document: Document) -> dict:
    return {
        "id": document.id,
        "filename": document.filename,
        "page_count": document.page_count,
        "status": document.status,
    }


def _collection_payload(db: Session, collection: Collection) -> dict:
    documents = db.scalars(
        select(Document)
        .join(CollectionDocument, CollectionDocument.document_id == Document.id)
        .where(CollectionDocument.collection_id == collection.id)
        .order_by(Document.created_at)
    ).all()
    return {
        "id": collection.id,
        "name": collection.name,
        "created_at": collection.created_at,
        "updated_at": collection.updated_at,
        "documents": [_document_summary(document) for document in documents],
    }


def list_collections(db: Session) -> list[dict]:
    collections = db.scalars(select(Collection).order_by(Collection.created_at)).all()
    return [_collection_payload(db, collection) for collection in collections]


def get_collection(db: Session, collection_id: str) -> dict:
    collection = db.get(Collection, collection_id)
    if not collection:
        raise CollectionNotFound(collection_id)
    return _collection_payload(db, collection)


def create_collection(db: Session, name: str) -> dict:
    collection = Collection(name=name)
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return _collection_payload(db, collection)


def rename_collection(db: Session, collection_id: str, name: str) -> dict:
    collection = db.get(Collection, collection_id)
    if not collection:
        raise CollectionNotFound(collection_id)
    collection.name = name
    db.commit()
    db.refresh(collection)
    return _collection_payload(db, collection)


def delete_collection(db: Session, collection_id: str) -> None:
    """Removes the collection and its membership rows only -- the underlying
    Document rows (and their chunks/vectors/uploaded files) are never touched
    here, per "deleting a collection should not automatically delete its
    documents".
    """
    collection = db.get(Collection, collection_id)
    if not collection:
        raise CollectionNotFound(collection_id)
    for membership in db.scalars(select(CollectionDocument).where(CollectionDocument.collection_id == collection_id)).all():
        db.delete(membership)
    db.delete(collection)
    db.commit()


def add_document(db: Session, collection_id: str, document_id: str) -> dict:
    collection = db.get(Collection, collection_id)
    if not collection:
        raise CollectionNotFound(collection_id)
    if not db.get(Document, document_id):
        raise DocumentNotFound(document_id)
    existing = db.scalar(
        select(CollectionDocument).where(CollectionDocument.collection_id == collection_id, CollectionDocument.document_id == document_id)
    )
    if not existing:
        db.add(CollectionDocument(collection_id=collection_id, document_id=document_id))
        db.commit()
    return _collection_payload(db, collection)


def remove_document(db: Session, collection_id: str, document_id: str) -> dict:
    """Removes membership only -- the document itself is left completely
    intact, per "Do NOT delete the underlying document when removing it from
    a collection".
    """
    collection = db.get(Collection, collection_id)
    if not collection:
        raise CollectionNotFound(collection_id)
    membership = db.scalar(
        select(CollectionDocument).where(CollectionDocument.collection_id == collection_id, CollectionDocument.document_id == document_id)
    )
    if membership:
        db.delete(membership)
        db.commit()
    return _collection_payload(db, collection)


def member_document_ids(db: Session, collection_id: str) -> set[str]:
    return set(db.scalars(select(CollectionDocument.document_id).where(CollectionDocument.collection_id == collection_id)).all())
