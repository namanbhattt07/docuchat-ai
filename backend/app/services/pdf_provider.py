import re
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import RLock

import fitz

from app.core.config import Settings, get_settings
from app.services.document_model import (
    PAGE_EMPTY,
    PAGE_FAILED,
    PAGE_NEEDS_OCR,
    PAGE_OCR,
    PAGE_TEXT,
    SOURCE_OCR,
    SOURCE_TEXT,
    BBox,
    DocumentBlock,
    DocumentPage,
    DocumentUnreadable,
    ImageRegion,
    PageBlock,
    ScannedDocument,
    UnsupportedDocumentType,
)
from app.services.ocr import OcrEngine, OcrLine, get_ocr_engine

# The PDF implementation of DocumentProvider (services/document_model.py) --
# the only place that knows about PyMuPDF. Ingestion, the citation/search
# caches, the suggested-questions backfill and page rendering all reach a PDF
# through here, so nothing outside this module parses one directly.
#
# COORDINATES: every bbox this module returns is in the page's *native,
# unrotated* space -- the space PyMuPDF reports text and image boxes in, and
# the space already stored for chunks and highlights. Rendering (which is
# rotation-aware) converts at the boundary: OCR pixel boxes go through the
# page's derotation matrix on the way in, and preview clips through its
# rotation matrix on the way out.

# PyMuPDF/MuPDF is not safe to drive from several threads at once. Ingestion
# runs its PDF work on a worker thread while the API can be rendering a page
# preview on another, so every MuPDF touch is serialized here. The lock is
# never held while OCR inference runs (only while opening/rendering).
_MUPDF_LOCK = RLock()

_FIGURE_CAPTION_PATTERN = re.compile(
    r"^(?P<kind>Figure|Fig\.?|Chart|Diagram|Graph|Plot)\s+(?P<number>\d+(?:[.\-]\d+)*[a-z]?)\s*[.:–—\-]\s*\S",
    re.IGNORECASE,
)
_CAPTION_MAX_CHARS = 300
_CANONICAL_CAPTION_KIND = {"fig": "Figure", "figure": "Figure", "chart": "Chart", "diagram": "Diagram", "graph": "Graph", "plot": "Plot"}


def extract_page_blocks(page: "fitz.Page") -> tuple[str, list[PageBlock]]:
    """Walk a page's text as layout blocks (PyMuPDF's structured dict
    output) rather than one flat string, normalize whitespace per block,
    and stitch them back into a single string while recording each block's
    character offsets, bounding box, and whether it looks like a heading
    (short line, meaningfully larger font than the page's median).

    This is what lets a chunk later be traced back to a location on the
    page (for highlighting the source in the viewer) and to the heading it
    most likely falls under (a "section" label), instead of just a bare
    page number.
    """
    text_dict = page.get_text("dict")
    sizes: list[float] = []
    raw_blocks: list[tuple[tuple[float, float, float, float], str, float]] = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # skip image blocks
            continue
        line_texts: list[str] = []
        max_size = 0.0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            if line_text.strip():
                line_texts.append(line_text)
            for span in spans:
                size = float(span.get("size", 0.0))
                sizes.append(size)
                max_size = max(max_size, size)
        block_text = re.sub(r"\s+", " ", " ".join(line_texts)).strip()
        if not block_text:
            continue
        raw_blocks.append((tuple(block["bbox"]), block_text, max_size))

    median_size = statistics.median(sizes) if sizes else 0.0

    page_text_parts: list[str] = []
    blocks: list[PageBlock] = []
    offset = 0
    for bbox, block_text, max_size in raw_blocks:
        if page_text_parts:
            page_text_parts.append(" ")
            offset += 1
        start = offset
        page_text_parts.append(block_text)
        offset += len(block_text)
        end = offset
        is_heading = bool(median_size) and max_size >= median_size * 1.25 and len(block_text) <= 120
        blocks.append(PageBlock(block_text, bbox, start, end, is_heading, font_size=max_size))

    return "".join(page_text_parts), blocks


def classify_page(native_chars: int, image_coverage: float, *, min_chars: int, min_coverage: float) -> tuple[str, str | None]:
    """The textless-page decision, kept as a pure function so the rule is
    testable on its own. In order:

    1. Enough native text  -> a text page. Never OCR'd, whatever images it has.
    2. Thin text but images cover a large share of the page -> a scanned page:
       route to OCR. Having *an* image is not enough; it has to dominate.
    3. Thin but non-empty text (a chapter divider, a page number) -> a text page.
    4. Nothing -> genuinely empty (blank, or only tiny decorative graphics).
    """
    if native_chars >= min_chars:
        return PAGE_TEXT, None
    if image_coverage >= min_coverage:
        return PAGE_NEEDS_OCR, None
    if native_chars > 0:
        return PAGE_TEXT, None
    if image_coverage > 0:
        return PAGE_EMPTY, "No readable text: the page only has small graphics."
    return PAGE_EMPTY, "Blank page."


def _native_char_count(text: str) -> int:
    return sum(1 for char in text if char.isalnum())


def _page_dimensions(page: "fitz.Page") -> fitz.Rect:
    """The page rect in native (unrotated) space."""
    rect = fitz.Rect(page.rect) * page.derotation_matrix
    rect.normalize()
    return rect


def _raster_infos(page: "fitz.Page") -> list[dict]:
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:  # noqa: BLE001 -- a damaged image object must not fail the whole page
        return []
    return [
        {"bbox": tuple(info["bbox"]), "xref": info.get("xref") or None, "width": info.get("width"), "height": info.get("height")}
        for info in infos
    ]


def _coverage(page_rect: fitz.Rect, boxes: list[tuple[float, float, float, float]]) -> float:
    area = page_rect.width * page_rect.height
    if area <= 0:
        return 0.0
    covered = 0.0
    for box in boxes:
        visible = fitz.Rect(box) & page_rect
        if not visible.is_empty:
            covered += visible.width * visible.height
    return min(1.0, covered / area)


def figure_captions(page_number: int, blocks: list[DocumentBlock], start_index: int) -> list[ImageRegion]:
    """"Figure 3: ..." caption blocks. Recorded so a figure can be found and
    cited by its label even when it is vector artwork (which has no raster
    image to detect); `kind="figure_caption"` says exactly what was detected.
    The separator after the number is required, so a sentence that merely
    starts "Figure 3 shows ..." is not mistaken for a caption.
    """
    found: list[ImageRegion] = []
    seen: set[str] = set()
    for block in blocks:
        match = _FIGURE_CAPTION_PATTERN.match(block.text)
        if not match:
            continue
        kind = _CANONICAL_CAPTION_KIND[match.group("kind").lower().rstrip(".")]
        label = f"{kind} {match.group('number')}"
        if label.lower() in seen:
            continue
        seen.add(label.lower())
        found.append(ImageRegion(
            page_number=page_number,
            image_index=start_index + len(found),
            kind="figure_caption",
            bbox=tuple(block.bbox),
            label=label,
            caption=block.text[:_CAPTION_MAX_CHARS],
        ))
    return found


def _assemble(parts: list[tuple[str, BBox, bool, float, str, float | None]]) -> tuple[str, list[DocumentBlock]]:
    """Join (text, bbox, is_heading, font_size, source_type, confidence)
    parts into one page string with the same single-space block separator
    extract_page_blocks uses, so offsets mean the same thing for native and
    OCR text."""
    pieces: list[str] = []
    blocks: list[DocumentBlock] = []
    offset = 0
    for text, bbox, is_heading, font_size, source_type, confidence in parts:
        if pieces:
            pieces.append(" ")
            offset += 1
        start = offset
        pieces.append(text)
        offset += len(text)
        blocks.append(DocumentBlock(text, bbox, start, offset, is_heading, font_size=font_size, source_type=source_type, confidence=confidence))
    return "".join(pieces), blocks


class PdfDocumentProvider:
    kind = "pdf"

    def __init__(self, ocr_engine: OcrEngine | None = None, settings: Settings | None = None) -> None:
        self._ocr_engine = ocr_engine
        self._settings = settings

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    @property
    def ocr_engine(self) -> OcrEngine:
        return self._ocr_engine or get_ocr_engine()

    # -- scan -----------------------------------------------------------------

    def scan(self, path: Path) -> ScannedDocument:
        settings = self.settings
        raw: list[dict] = []
        with _MUPDF_LOCK:
            try:
                pdf = fitz.open(path)
            except Exception as exc:  # noqa: BLE001 -- MuPDF raises several unrelated types for a damaged file
                raise DocumentUnreadable("This file couldn't be opened as a PDF. It may be damaged or not a real PDF.") from exc
            try:
                if pdf.needs_pass:
                    raise DocumentUnreadable("This PDF is password-protected. Remove the password, then upload it again.")
                page_count = len(pdf)
                if page_count == 0:
                    # MuPDF "repairs" some damaged files into an empty document; that is a
                    # broken file, not a blank one, and must not read as "no text".
                    raise DocumentUnreadable("This PDF has no readable pages. The file may be damaged.")
                native_toc = pdf.get_toc()
                for page in pdf:
                    text, blocks = extract_page_blocks(page)
                    rect = _page_dimensions(page)
                    raw.append({"text": text, "blocks": blocks, "rect": rect, "rasters": _raster_infos(page)})
            finally:
                pdf.close()

        # An image object that recurs across most pages is a running
        # header/logo/letterhead, not content: don't count it toward "this
        # page is a scan" and don't report it as a figure.
        repeats: Counter = Counter()
        for entry in raw:
            repeats.update({info["xref"] for info in entry["rasters"] if info["xref"]})
        decorative_xrefs = {xref for xref, count in repeats.items() if page_count >= 4 and count > page_count / 2}

        pages: list[DocumentPage] = []
        images: list[ImageRegion] = []
        for index, entry in enumerate(raw):
            page_number = index + 1
            rect: fitz.Rect = entry["rect"]
            content_rasters = [info for info in entry["rasters"] if info["xref"] not in decorative_xrefs]
            status, detail = classify_page(
                _native_char_count(entry["text"]),
                _coverage(rect, [info["bbox"] for info in content_rasters]),
                min_chars=settings.ocr_min_native_chars,
                min_coverage=settings.ocr_min_image_coverage,
            )
            page = DocumentPage(
                page_number=page_number,
                text=entry["text"],
                blocks=entry["blocks"],
                status=status,
                status_detail=detail,
                width=round(rect.width, 2),
                height=round(rect.height, 2),
            )
            page.images = self._page_figures(page, rect, content_rasters, settings)
            images.extend(page.images)
            pages.append(page)
        return ScannedDocument(page_count=page_count, pages=pages, native_toc=native_toc, images=images)

    @staticmethod
    def _page_figures(page: DocumentPage, rect: fitz.Rect, rasters: list[dict], settings: Settings) -> list[ImageRegion]:
        page_area = rect.width * rect.height
        figures: list[ImageRegion] = []
        # A page-sized image is a scan/background, not a figure. On a page
        # still awaiting OCR it is kept for now: if OCR then finds no text at
        # all, the page is an image page (photo/artwork) and the image is the
        # content (see _drop_page_sized_images for when it is discarded).
        keep_page_sized = page.status == PAGE_NEEDS_OCR
        for info in rasters:
            box = fitz.Rect(info["bbox"]) & rect
            if box.is_empty:
                continue
            width_px = info["width"] or int(box.width)
            height_px = info["height"] or int(box.height)
            too_small = min(width_px, height_px) < settings.image_min_dimension_px
            too_little_page = page_area > 0 and (box.width * box.height) / page_area < settings.image_min_area_ratio
            page_sized = page_area > 0 and (box.width * box.height) / page_area >= settings.image_max_area_ratio
            if too_small or too_little_page or (page_sized and not keep_page_sized):
                continue
            figures.append(ImageRegion(
                page_number=page.page_number,
                image_index=len(figures),
                kind="image",
                bbox=(round(box.x0, 2), round(box.y0, 2), round(box.x1, 2), round(box.y1, 2)),
                width=int(width_px),
                height=int(height_px),
                xref=info["xref"],
            ))
        figures.extend(figure_captions(page.page_number, page.blocks, start_index=len(figures)))
        return figures

    # -- OCR ------------------------------------------------------------------

    def ocr_availability(self) -> tuple[bool, str | None]:
        if not self.settings.ocr_enabled:
            return False, "OCR is disabled (OCR_ENABLED=false)."
        available, reason = self.ocr_engine.availability()
        return available, None if available else (reason or "OCR engine unavailable.")

    def recognize_page(self, path: Path, page: DocumentPage) -> DocumentPage:
        settings = self.settings
        available, reason = self.ocr_availability()
        if not available:
            return self._failed(page, reason or "OCR engine unavailable.")

        try:
            with _MUPDF_LOCK:
                pdf = fitz.open(path)
                try:
                    source = pdf[page.page_number - 1]
                    longest = max(source.rect.width, source.rect.height) or 1.0
                    zoom = min(settings.ocr_render_dpi / 72, settings.ocr_max_image_px / longest)
                    pixmap = source.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
                    png = pixmap.tobytes("png")
                    # Exact per-axis scale (rounding of the pixmap size means
                    # it is not precisely `zoom`) from image px back to points.
                    scale_x = pixmap.width / source.rect.width
                    scale_y = pixmap.height / source.rect.height
                    derotation = fitz.Matrix(source.derotation_matrix)
                finally:
                    pdf.close()
            lines = self.ocr_engine.recognize(png)
        except Exception as exc:  # noqa: BLE001 -- any engine/render failure is reported per page, never fatal to the document
            return self._failed(page, f"OCR failed on this page: {exc}"[:280])

        return self._page_from_ocr(page, lines, scale_x, scale_y, derotation)

    def _page_from_ocr(self, page: DocumentPage, lines: list[OcrLine], scale_x: float, scale_y: float, derotation: fitz.Matrix) -> DocumentPage:
        settings = self.settings
        kept = [line for line in lines if re.sub(r"\s+", " ", line.text).strip() and line.confidence >= settings.ocr_min_confidence]
        if not kept:
            if page.text.strip():
                # OCR added nothing, but there is still real (if thin) native text.
                return replace(page, status=PAGE_TEXT, status_detail="OCR found no additional text.", images=self._drop_page_sized_images(page))
            return replace(page, status=PAGE_EMPTY, status_detail="Image-only page: OCR found no readable text.")

        heights = [(line.bbox[3] - line.bbox[1]) / scale_y for line in kept]
        median_height = statistics.median(heights)
        ocr_parts: list[tuple[str, BBox, bool, float, str, float | None]] = []
        for line, height in zip(kept, heights):
            text = re.sub(r"\s+", " ", line.text).strip()
            box = fitz.Rect(line.bbox[0] / scale_x, line.bbox[1] / scale_y, line.bbox[2] / scale_x, line.bbox[3] / scale_y) * derotation
            box.normalize()
            is_heading = median_height > 0 and height >= median_height * 1.25 and len(text) <= 120
            ocr_parts.append((text, (round(box.x0, 2), round(box.y0, 2), round(box.x1, 2), round(box.y1, 2)), is_heading, round(height, 2), SOURCE_OCR, round(line.confidence, 3)))

        # Keep any thin native text (a page stamp, a scanner's footer) that OCR
        # did not also read from the image, rather than silently dropping it.
        ocr_text = " ".join(part[0] for part in ocr_parts).lower()
        native_parts = [
            (block.text, tuple(block.bbox), block.is_heading, block.font_size, SOURCE_TEXT, None)
            for block in page.blocks
            if block.text.lower() not in ocr_text
        ]
        text, blocks = _assemble(native_parts + ocr_parts)
        confidence = round(sum(line.confidence for line in kept) / len(kept), 3)
        # Captions on a scanned page only exist as OCR text, so figure labels
        # ("Figure 3: ...") are detected from the OCR'd blocks too.
        remaining = self._drop_page_sized_images(page)
        captions = figure_captions(page.page_number, blocks, start_index=len(remaining))
        return replace(
            page, text=text, blocks=blocks, status=PAGE_OCR, source_type=SOURCE_OCR, status_detail=None,
            ocr_confidence=confidence, images=[*remaining, *captions],
        )

    def _drop_page_sized_images(self, page: DocumentPage) -> list[ImageRegion]:
        """Once a page is known to be a scan (OCR'd, or failed), its full-page
        image is the scan itself, not a figure."""
        area = page.width * page.height
        limit = self.settings.image_max_area_ratio
        return [
            image for image in page.images
            if not (image.kind == "image" and area > 0 and (image.bbox[2] - image.bbox[0]) * (image.bbox[3] - image.bbox[1]) / area >= limit)
        ]

    def _failed(self, page: DocumentPage, reason: str) -> DocumentPage:
        return replace(page, status=PAGE_FAILED, status_detail=reason[:300], images=self._drop_page_sized_images(page))

    # -- rendering / metadata -------------------------------------------------

    def render_page(self, path: Path, page_number: int, *, dpi: int, clip: BBox | None = None) -> bytes:
        with _MUPDF_LOCK:
            pdf = fitz.open(path)
            try:
                if not 1 <= page_number <= len(pdf):
                    raise IndexError(f"Page {page_number} is out of range (document has {len(pdf)} pages).")
                page = pdf[page_number - 1]
                region = None
                if clip is not None:
                    # Stored boxes are unrotated; get_pixmap wants rotated space.
                    region = fitz.Rect(clip) * page.rotation_matrix
                    region.normalize()
                    region = region & page.rect
                    if region.is_empty:
                        region = None
                return page.get_pixmap(dpi=dpi, clip=region, alpha=False).tobytes("png")
            finally:
                pdf.close()

    def list_images(self, path: Path) -> list[ImageRegion]:
        return self.scan(path).images


_PROVIDERS_BY_SUFFIX: dict[str, type] = {".pdf": PdfDocumentProvider}


def document_provider_for(path: Path | str) -> PdfDocumentProvider:
    """PDF is the only real implementation. This is the single seam a future
    format would register at (add its suffix here) -- deliberately not a
    plugin system."""
    suffix = Path(path).suffix.lower()
    provider_type = _PROVIDERS_BY_SUFFIX.get(suffix)
    if provider_type is None:
        raise UnsupportedDocumentType(f"No document provider for '{suffix or 'this file type'}'; only PDF is supported.")
    return provider_type()
