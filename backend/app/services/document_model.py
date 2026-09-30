from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

# Group 7 document abstraction: Document -> Pages -> Blocks -> Chunks, with no
# dependency on any file format. PDF (services/pdf_provider.py) is the only
# concrete implementation; these types are what ingestion, chunking, search
# and the citation code already consumed under other names (`PageBlock`,
# `PageIndexEntry`) -- those names remain as aliases so nothing that imports
# them has to change, and no stored record changes shape.

# Per-page processing outcome. Persisted (page_text.status) so the UI and
# API report it explicitly instead of inferring it from empty text later.
PAGE_TEXT = "text"        # usable native text layer
PAGE_OCR = "ocr"          # no usable text layer; text recovered by OCR
PAGE_EMPTY = "empty"      # genuinely nothing to read (blank, or an image OCR found no text in)
PAGE_FAILED = "failed"    # needed OCR and it could not be done
# Transient (scan -> OCR hand-off inside ingestion only; never persisted).
PAGE_NEEDS_OCR = "needs_ocr"

SOURCE_TEXT = "text"
SOURCE_OCR = "ocr"

BBox = tuple[float, float, float, float]


@dataclass
class DocumentBlock:
    """One contiguous run of page text with its location. `start`/`end` are
    character offsets into the owning page's assembled text."""

    text: str
    bbox: BBox
    start: int
    end: int
    is_heading: bool
    # Largest font size in the block (native) or line height in points (OCR).
    # Only used to bucket heading levels for the heuristic TOC.
    font_size: float = 0.0
    source_type: str = SOURCE_TEXT
    confidence: float | None = None


# Historical name, still imported by ingestion code and tests.
PageBlock = DocumentBlock


@dataclass
class ImageRegion:
    """A figure-like element on a page. Metadata only -- never pixel data;
    `xref` is the reference back into the source PDF and previews are
    rendered on demand (see PdfDocumentProvider.render_page)."""

    page_number: int
    image_index: int
    kind: str  # "image" (embedded raster) | "figure_caption" (a "Figure N" caption line)
    bbox: BBox
    width: int | None = None
    height: int | None = None
    xref: int | None = None
    label: str | None = None
    caption: str | None = None


@dataclass
class DocumentPage:
    page_number: int  # 1-based, always the page's real position in the file
    text: str
    blocks: list[DocumentBlock] = field(default_factory=list)
    status: str = PAGE_TEXT
    source_type: str = SOURCE_TEXT
    status_detail: str | None = None
    ocr_confidence: float | None = None
    width: float = 0.0
    height: float = 0.0
    images: list[ImageRegion] = field(default_factory=list)


# Historical name, still imported by ingestion code and tests.
PageIndexEntry = DocumentPage


@dataclass
class DocumentChunk:
    """A retrievable slice of one page. Maps 1:1 onto the existing `Chunk`
    row (id, page_number, content, section, offsets, bbox, source_type)."""

    page_number: int
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    bbox: list[float] | None
    section: str | None
    source_type: str = SOURCE_TEXT

    def chunk_id(self, document_id: str) -> str:
        return f"{document_id}-{self.page_number}-{self.chunk_index}"


@dataclass
class ScannedDocument:
    """Everything one pass over a file yields, before any OCR: pages (with
    `needs_ocr` ones flagged), the file's own outline, and figure metadata."""

    page_count: int
    pages: list[DocumentPage]
    native_toc: list = field(default_factory=list)
    images: list[ImageRegion] = field(default_factory=list)


class UnsupportedDocumentType(ValueError):
    pass


class DocumentUnreadable(ValueError):
    """The file can't be read at all (password-protected, damaged, no pages).
    The message is written for the person who uploaded it and is stored as the
    document's `status_detail`, so it says what to do about it."""


@runtime_checkable
class DocumentProvider(Protocol):
    """A file format's implementation of the Document -> Pages -> Blocks
    contract. Chunking, embedding, indexing, retrieval and citations are all
    format-independent and operate on what this returns.

    The methods are synchronous and CPU/IO bound; async callers run them via
    `asyncio.to_thread`.
    """

    kind: str

    def scan(self, path: Path) -> ScannedDocument:
        """Native extraction + classification of every page. Never runs OCR."""
        ...

    def ocr_availability(self) -> tuple[bool, str | None]:
        """(can OCR run right now, reason if not) -- checked once per
        document so an unavailable engine fails fast with one clear message."""
        ...

    def recognize_page(self, path: Path, page: DocumentPage) -> DocumentPage:
        """Resolve one `PAGE_NEEDS_OCR` page to `ocr` / `empty` / `failed`."""
        ...

    def render_page(self, path: Path, page_number: int, *, dpi: int, clip: BBox | None = None) -> bytes:
        """PNG bytes of one page (or a clip of it, in unrotated page space)."""
        ...

    def list_images(self, path: Path) -> list[ImageRegion]:
        """Figure metadata only (no full text extraction) -- used to backfill
        documents ingested before figure metadata existed."""
        ...
