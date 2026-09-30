from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.services import collections as collections_service

router = APIRouter(prefix="/collections", tags=["collections"])


class CreateCollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class RenameCollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class AddDocumentRequest(BaseModel):
    document_id: str


@router.get("")
def list_collections(db: Session = Depends(get_db)):
    return collections_service.list_collections(db)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_collection(request: CreateCollectionRequest, db: Session = Depends(get_db)):
    return collections_service.create_collection(db, request.name.strip())


@router.get("/{collection_id}")
def get_collection(collection_id: str, db: Session = Depends(get_db)):
    try:
        return collections_service.get_collection(db, collection_id)
    except collections_service.CollectionNotFound as exc:
        raise HTTPException(status_code=404, detail="Collection not found.") from exc


@router.patch("/{collection_id}")
def rename_collection(collection_id: str, request: RenameCollectionRequest, db: Session = Depends(get_db)):
    try:
        return collections_service.rename_collection(db, collection_id, request.name.strip())
    except collections_service.CollectionNotFound as exc:
        raise HTTPException(status_code=404, detail="Collection not found.") from exc


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: str, db: Session = Depends(get_db)):
    try:
        collections_service.delete_collection(db, collection_id)
    except collections_service.CollectionNotFound as exc:
        raise HTTPException(status_code=404, detail="Collection not found.") from exc


@router.post("/{collection_id}/documents")
def add_document(collection_id: str, request: AddDocumentRequest, db: Session = Depends(get_db)):
    try:
        return collections_service.add_document(db, collection_id, request.document_id)
    except collections_service.CollectionNotFound as exc:
        raise HTTPException(status_code=404, detail="Collection not found.") from exc
    except collections_service.DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc


@router.delete("/{collection_id}/documents/{document_id}")
def remove_document(collection_id: str, document_id: str, db: Session = Depends(get_db)):
    try:
        return collections_service.remove_document(db, collection_id, document_id)
    except collections_service.CollectionNotFound as exc:
        raise HTTPException(status_code=404, detail="Collection not found.") from exc
