import asyncio

from fakes import FakeEmbeddingProvider, FakeLLMProvider, FakeOcrEngine

from app.models import Document
from app.services.documents import process_document
from app.services.pdf_provider import PdfDocumentProvider
from app.services.providers import use_providers


class RecordingCollection:
    """Vector-store stand-in: records what ingestion embedded, returns no
    vector hits (keyword/FTS5 retrieval still runs for real)."""

    def __init__(self) -> None:
        self.added: list[dict] = []

    def add(self, ids, documents, embeddings, metadatas):
        self.added.append({"ids": ids, "documents": documents, "metadatas": metadatas})

    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


def ingest(db_session, path, engine=None, *, document_id="doc-1", filename="doc.pdf", collection=None):
    document = Document(id=document_id, filename=filename, status="processing")
    db_session.add(document)
    db_session.commit()
    collection = collection or RecordingCollection()
    provider = PdfDocumentProvider(ocr_engine=engine or FakeOcrEngine())
    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        asyncio.run(process_document(db_session, document, path, collection, provider=provider))
    return document, collection
