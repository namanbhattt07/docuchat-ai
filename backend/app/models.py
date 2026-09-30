from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(String(255))
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="processing")
    status_detail: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Cached table-of-contents (JSON list of {id, title, page, level, source}),
    # built once during ingestion (see services/documents.py::build_toc) so the
    # TOC endpoint never has to re-open/re-parse the PDF. Null means "not
    # indexed yet" (e.g. a document ingested before this column existed) --
    # the TOC/search endpoints lazily backfill it via ensure_document_index.
    toc_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Cached ~5 document-aware suggested questions (JSON list[str]), generated
    # once during ingestion (see services/suggestions.py) so the chat empty
    # state never has to call the LLM on page load. Null means "not
    # generated yet" (older documents, or generation failed) -- callers treat
    # that the same as an empty list rather than erroring.
    suggested_questions_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Group 7: 1 once figure/image metadata (document_images) has been
    # recorded for this document -- at ingestion for new documents, lazily on
    # first request for ones that pre-date it (see services/visuals.py). The
    # flag is what tells "scanned, no figures" apart from "never scanned".
    visuals_indexed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # Page-aware metadata, captured during PyMuPDF extraction (see
    # services/documents.py::extract_page_blocks) so a citation can be
    # traced back to exactly where on the page it came from.
    section: Mapped[str | None] = mapped_column(String(300), nullable=True)
    start_offset: Mapped[int] = mapped_column(Integer, default=0)
    end_offset: Mapped[int] = mapped_column(Integer, default=0)
    # [x0, y0, x1, y1] in PDF point space (page.rect), or null when a chunk
    # spans no locatable text block (shouldn't normally happen, but a
    # renderer must not assume this is always present).
    bbox: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    source_type: Mapped[str] = mapped_column(String(20), default="text")


class PageText(Base):
    """One row per PDF page: the same normalized text/blocks that
    extract_page_blocks already produces during ingestion, persisted so the
    in-document keyword search and citation passage-locating fallback (see
    services/search.py, services/citations.py) never have to re-open and
    re-parse the PDF. `blocks_json` is a list of {start, end, bbox} spans
    used to map a matched text range back to a bounding box on the page.
    """
    __tablename__ = "page_text"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text, default="")
    blocks_json: Mapped[str] = mapped_column(Text, default="[]")
    # Group 7 explicit per-page processing outcome (see document_model.PAGE_*):
    # "text" | "ocr" | "empty" | "failed". Null only for rows written before
    # this column existed -- those pages were never classified, and the API
    # reports them as "unknown" rather than guessing from their text.
    status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    source_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status_detail: Mapped[str | None] = mapped_column(String(300), nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class DocumentImage(Base):
    """Group 7: metadata for one figure-like element on a page -- an embedded
    raster image, or a "Figure N" caption (`kind`). Deliberately holds no
    pixel data: `xref` points back into the source PDF and previews are
    rendered on demand into data/processed (see services/visuals.py), so the
    database never grows with image size.
    """
    __tablename__ = "document_images"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer, index=True)
    image_index: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(20), default="image")
    label: Mapped[str | None] = mapped_column(String(60), nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    # [x0, y0, x1, y1] in native (unrotated) PDF point space, same as Chunk.bbox.
    bbox: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    xref: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Group 5 threads/follow-ups: which prior message (usually an assistant
    # turn) the user explicitly hit "Reply" on, so a follow-up can pull that
    # turn's context back into history even if the conversation has moved on
    # since (see api/v1/chat.py::_recent_history's anchor_message_id param).
    # No FK constraint -- SQLite FK enforcement is off by default here and a
    # dangling reference (the target turn was never deleted independently,
    # but keeping this loose avoids ON DELETE ordering issues) should never
    # block sending a new message.
    reply_to_message_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Collection(Base):
    """Group 5: a named grouping of documents the user chats with as a single
    workspace (see MULTI-DOCUMENT COLLECTIONS). Deliberately holds no
    document list itself -- membership lives in CollectionDocument so a
    document can belong to more than one collection and deleting a
    collection never touches the underlying Document rows.
    """
    __tablename__ = "collections"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CollectionDocument(Base):
    """Many-to-many membership row between Collection and Document. A plain
    join table (not a list column on either side) so a document can sit in
    multiple collections and removing it from one never risks the others.
    """
    __tablename__ = "collection_documents"
    __table_args__ = (UniqueConstraint("collection_id", "document_id", name="uq_collection_document"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PromptTemplate(Base):
    """Group 7: a user-defined prompt template (name / description /
    instruction, plus an optional output format). Built-in templates are code
    constants (services/templates.py), not rows here, so upgrading the app
    never has to migrate or overwrite anything a user saved.
    """
    __tablename__ = "prompt_templates"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(String(300), default="")
    instruction: Mapped[str] = mapped_column(Text)
    output_format: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
