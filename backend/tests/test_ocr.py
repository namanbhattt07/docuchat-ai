import asyncio
import json

import fitz
import numpy as np
import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider, FakeOcrEngine
from ingest import RecordingCollection, ingest
from pdfs import LOREM, add_blank_page, add_logo_text_page, add_scanned_page, add_text_page, build_pdf
from sqlalchemy import select

import app.api.v1.chat as chat_module
import app.services.documents as documents_module
from app.api.v1.chat import ChatRequest, ask, build_citations, format_source_block
from app.core.config import get_settings
from app.models import Chunk, Document, PageText
from app.services import keyword_index
from app.services.document_model import PAGE_EMPTY, PAGE_FAILED, PAGE_NEEDS_OCR, PAGE_OCR, PAGE_TEXT
from app.services.documents import process_document
from app.services.learning import sample_document_sources
from app.services.ocr import OCR_SETUP_MESSAGE, OcrLine, RapidOcrEngine
from app.services.pdf_provider import PdfDocumentProvider, classify_page, document_provider_for
from app.services.providers import use_providers
from app.services.retrieval import retrieve_overview, retrieve_with_plan
from app.services.retrieval_types import RetrievalMode, RetrievalPlan

# ---------------------------------------------------------------------------
# Group 7 / 26: OCR fallback for scanned PDFs. Uses a scripted fake OCR engine
# for everything deterministic, plus a small number of tests against the real
# local engine (skipped, with the reason, if it isn't installed).
# ---------------------------------------------------------------------------

SCANNED_TEXT = "Scanned Page. The MQTT broker handles telemetry from every gateway. Battery life is 18 months."


def _ocr_line(text: str, box=(100.0, 200.0, 900.0, 260.0), confidence: float = 0.97) -> OcrLine:
    return OcrLine(text=text, bbox=box, confidence=confidence)


LONG_OCR_TEXT = (
    "The MQTT broker handles telemetry from every gateway in the building. "
    "Battery life of the wireless sensor nodes is 18 months under normal duty cycles."
)


def _mixed_pdf(path, scanned_text: str = SCANNED_TEXT):
    """1 native text | 2 blank | 3 scanned | 4 native text + small logo | 5 native text"""
    return build_pdf(path, [
        add_text_page,
        add_blank_page,
        lambda doc: add_scanned_page(doc, scanned_text),
        add_logo_text_page,
        lambda doc: add_text_page(doc, "Deep learning uses layered neural networks to model complex functions.", heading="Deep Learning"),
    ])


@pytest.fixture()
def settings():
    return get_settings()


# ---------- A. textless page detection ----------


@pytest.mark.parametrize(("native_chars", "coverage", "expected"), [
    (500, 0.0, PAGE_TEXT),            # ordinary text page
    (500, 1.0, PAGE_TEXT),            # lots of text AND a full-page image: never OCR'd
    (10, 0.9, PAGE_NEEDS_OCR),        # thin text under a page-sized image: scanned
    (0, 0.9, PAGE_NEEDS_OCR),
    (10, 0.05, PAGE_TEXT),            # thin text, small image: a chapter divider, not a scan
    (0, 0.05, PAGE_EMPTY),            # only a tiny graphic: nothing to read
    (0, 0.0, PAGE_EMPTY),             # genuinely blank
])
def test_classify_page_rules(native_chars, coverage, expected) -> None:
    status, _ = classify_page(native_chars, coverage, min_chars=30, min_coverage=0.5)
    assert status == expected


def test_having_an_image_alone_does_not_make_a_page_scanned() -> None:
    assert classify_page(200, 0.6, min_chars=30, min_coverage=0.5)[0] == PAGE_TEXT


def test_blank_page_detail_says_blank() -> None:
    assert classify_page(0, 0.0, min_chars=30, min_coverage=0.5) == (PAGE_EMPTY, "Blank page.")


def test_scan_classifies_each_page_kind(tmp_path) -> None:
    path = _mixed_pdf(tmp_path / "mixed.pdf")

    scan = PdfDocumentProvider(ocr_engine=FakeOcrEngine()).scan(path)

    assert scan.page_count == 5
    assert [page.status for page in scan.pages] == [PAGE_TEXT, PAGE_EMPTY, PAGE_NEEDS_OCR, PAGE_TEXT, PAGE_TEXT]
    assert [page.page_number for page in scan.pages] == [1, 2, 3, 4, 5]
    assert scan.pages[1].status_detail == "Blank page."
    assert scan.pages[2].text == ""  # scanned page has no text layer
    # the logo on page 4 is decorative: not a figure, and not a reason to OCR
    assert scan.pages[3].images == []


def test_a_repeated_letterhead_image_is_not_treated_as_a_scan(tmp_path) -> None:
    """The same full-page background on every page (letterhead/watermark) must
    not send textless pages to OCR nor be reported as a figure."""
    doc = fitz.open()
    background = fitz.open()
    source_page = background.new_page(width=595, height=842)
    source_page.draw_rect(source_page.rect, color=None, fill=(0.95, 0.95, 0.9))
    png = source_page.get_pixmap(dpi=40).tobytes("png")
    xref = 0
    for index in range(6):
        page = add_text_page(doc, LOREM) if index % 2 == 0 else doc.new_page()
        xref = page.insert_image(page.rect, stream=png) if not xref else page.insert_image(page.rect, xref=xref)
    path = tmp_path / "letterhead.pdf"
    doc.save(path)

    scan = PdfDocumentProvider(ocr_engine=FakeOcrEngine()).scan(path)

    assert [page.status for page in scan.pages] == [PAGE_TEXT, PAGE_EMPTY] * 3
    assert scan.images == []


# ---------- B. OCR fallback: recognize a page ----------


def _scanned_page(tmp_path, text=SCANNED_TEXT):
    path = build_pdf(tmp_path / "scan.pdf", [lambda doc: add_scanned_page(doc, text)])
    provider_for_scan = PdfDocumentProvider(ocr_engine=FakeOcrEngine())
    return path, provider_for_scan.scan(path).pages[0]


def test_ocr_success_converts_lines_to_blocks_in_page_points(tmp_path) -> None:
    path, page = _scanned_page(tmp_path)
    engine = FakeOcrEngine(lambda png: [
        # A box covering x 10%-60%, y 20%-25% of the rendered bitmap.
        _ocr_line("The MQTT broker handles telemetry.", box=_fractional_box(png, 0.10, 0.20, 0.60, 0.25)),
    ])

    result = PdfDocumentProvider(ocr_engine=engine).recognize_page(path, page)

    assert result.status == PAGE_OCR and result.source_type == "ocr"
    assert result.page_number == 1
    assert result.text == "The MQTT broker handles telemetry."
    block = result.blocks[0]
    assert block.source_type == "ocr" and block.confidence == 0.97
    # 595 x 842 pt page: the pixel fractions must come back as point fractions.
    assert block.bbox == pytest.approx((0.10 * 595, 0.20 * 842, 0.60 * 595, 0.25 * 842), abs=1.0)
    assert (block.start, block.end) == (0, len(result.text))
    assert result.ocr_confidence == 0.97


def _fractional_box(png: bytes, x0: float, y0: float, x1: float, y1: float):
    pixmap = fitz.Pixmap(png)
    return (x0 * pixmap.width, y0 * pixmap.height, x1 * pixmap.width, y1 * pixmap.height)


def test_ocr_lines_below_the_confidence_floor_or_blank_are_dropped(tmp_path) -> None:
    path, page = _scanned_page(tmp_path)
    engine = FakeOcrEngine([
        _ocr_line("Reliable line about sensors.", confidence=0.95),
        _ocr_line("gArbLe n0ise", confidence=0.2),
        _ocr_line("   ", confidence=0.99),
    ])

    result = PdfDocumentProvider(ocr_engine=engine).recognize_page(path, page)

    assert result.text == "Reliable line about sensors."
    assert len(result.blocks) == 1


def test_ocr_marks_taller_lines_as_headings(tmp_path) -> None:
    path, page = _scanned_page(tmp_path)
    engine = FakeOcrEngine([
        _ocr_line("System Overview", box=(100, 100, 700, 220)),
        _ocr_line("The broker forwards telemetry to the cloud.", box=(100, 300, 1200, 360)),
        _ocr_line("Gateways buffer readings while offline.", box=(100, 380, 1200, 440)),
    ])

    result = PdfDocumentProvider(ocr_engine=engine).recognize_page(path, page)

    assert [block.is_heading for block in result.blocks] == [True, False, False]


def test_thin_native_text_is_kept_alongside_ocr_text(tmp_path) -> None:
    doc = fitz.open()
    page = add_scanned_page(doc, SCANNED_TEXT)
    page.insert_text((72, 30), "Scan 0042", fontsize=8)  # scanner stamp: 8 chars, below the threshold
    path = tmp_path / "stamped.pdf"
    doc.save(path)
    provider = PdfDocumentProvider(ocr_engine=FakeOcrEngine([_ocr_line("Battery life is 18 months.")]))
    scanned = provider.scan(path).pages[0]
    assert scanned.status == PAGE_NEEDS_OCR and "Scan 0042" in scanned.text

    result = provider.recognize_page(path, scanned)

    assert result.status == PAGE_OCR
    assert "Scan 0042" in result.text and "Battery life is 18 months." in result.text
    assert {block.source_type for block in result.blocks} == {"text", "ocr"}


def test_ocr_that_finds_nothing_on_an_image_page_is_empty_not_failed(tmp_path) -> None:
    path, page = _scanned_page(tmp_path, text="")
    result = PdfDocumentProvider(ocr_engine=FakeOcrEngine([])).recognize_page(path, page)
    assert result.status == PAGE_EMPTY
    assert "OCR found no readable text" in result.status_detail


def test_ocr_engine_failure_is_a_per_page_failed_status(tmp_path) -> None:
    path, page = _scanned_page(tmp_path)
    result = PdfDocumentProvider(ocr_engine=FakeOcrEngine(RuntimeError("model crashed"))).recognize_page(path, page)
    assert result.status == PAGE_FAILED
    assert "model crashed" in result.status_detail


def test_missing_ocr_engine_reports_the_setup_message(tmp_path) -> None:
    path, page = _scanned_page(tmp_path)
    engine = FakeOcrEngine(available=False, reason=OCR_SETUP_MESSAGE)
    result = PdfDocumentProvider(ocr_engine=engine).recognize_page(path, page)
    assert result.status == PAGE_FAILED
    assert "pip install rapidocr" in result.status_detail
    assert engine.calls == 0


def test_ocr_can_be_disabled_by_setting(tmp_path, monkeypatch, settings) -> None:
    monkeypatch.setattr(settings, "ocr_enabled", False)
    path, page = _scanned_page(tmp_path)
    engine = FakeOcrEngine([_ocr_line("text")])
    result = PdfDocumentProvider(ocr_engine=engine).recognize_page(path, page)
    assert result.status == PAGE_FAILED and "disabled" in result.status_detail
    assert engine.calls == 0


def test_rapidocr_reports_a_clear_setup_message_when_not_installed(monkeypatch) -> None:
    import app.services.ocr as ocr_module

    monkeypatch.setattr(ocr_module.importlib.util, "find_spec", lambda name: None)
    available, reason = RapidOcrEngine().availability()
    assert available is False
    assert reason == OCR_SETUP_MESSAGE and "pip install rapidocr" in reason


# ---------- C. page / coordinate mapping ----------


def _marker_pdf(path, rotation: int):
    """A 400x600pt page whose only content is a black rectangle at a known
    place in native (unrotated) coordinates, drawn into a page-sized image."""
    doc = fitz.open()
    source = fitz.open()
    source_page = source.new_page(width=400, height=600)
    source_page.draw_rect(fitz.Rect(60, 110, 260, 150), color=None, fill=(0, 0, 0))
    png = source_page.get_pixmap(dpi=72).tobytes("png")
    page = doc.new_page(width=400, height=600)
    page.insert_image(page.rect, stream=png)
    page.set_rotation(rotation)
    doc.save(path)
    return path


def _ink_box(png: bytes) -> tuple[float, float, float, float]:
    pixmap = fitz.Pixmap(png)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)[:, :, 0]
    ys, xs = np.where(pixels < 128)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_ocr_boxes_land_in_native_page_space_even_on_rotated_pages(tmp_path, rotation) -> None:
    """Native text/image boxes are stored in unrotated page space; OCR boxes
    come from a rotation-aware render. They must be mapped into the *same*
    space or highlights would land in the wrong place on rotated scans."""
    path = _marker_pdf(tmp_path / f"marker-{rotation}.pdf", rotation)
    provider = PdfDocumentProvider(ocr_engine=FakeOcrEngine(lambda png: [_ocr_line("marker", box=_ink_box(png))]))
    page = provider.scan(path).pages[0]
    assert page.status == PAGE_NEEDS_OCR

    result = provider.recognize_page(path, page)

    assert result.status == PAGE_OCR
    assert result.blocks[0].bbox == pytest.approx((60, 110, 260, 150), abs=2.5)


def test_the_page_image_is_capped_so_a_huge_page_cannot_allocate_a_huge_bitmap(tmp_path, monkeypatch, settings) -> None:
    monkeypatch.setattr(settings, "ocr_max_image_px", 500)
    path, page = _scanned_page(tmp_path)
    engine = FakeOcrEngine([_ocr_line("x")])
    PdfDocumentProvider(ocr_engine=engine).recognize_page(path, page)
    pixmap = fitz.Pixmap(engine.last_png)
    assert max(pixmap.width, pixmap.height) <= 501


# ---------- D. ingestion: status, indexing, page numbers ----------


def _ingest(db_session, path, engine, *, document_id="doc-1", collection=None, embedder=None, llm=None):
    return ingest(db_session, path, engine, document_id=document_id, filename="scan.pdf", collection=collection)


@pytest.fixture()
def fast_embedding(monkeypatch):
    async def fake_embed_in_batches(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", fake_embed_in_batches)


def _page_rows(db_session, document_id="doc-1"):
    return {row.page_number: row for row in db_session.scalars(select(PageText).where(PageText.document_id == document_id)).all()}


def test_ingestion_records_explicit_page_status_for_every_kind(db_session, tmp_path, fast_embedding) -> None:
    path = _mixed_pdf(tmp_path / "mixed.pdf")
    engine = FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)])

    document, _ = _ingest(db_session, path, engine)

    rows = _page_rows(db_session)
    assert {number: row.status for number, row in rows.items()} == {1: "text", 2: "empty", 3: "ocr", 4: "text", 5: "text"}
    assert rows[2].status_detail == "Blank page."
    assert rows[3].source_type == "ocr" and rows[3].ocr_confidence == 0.97
    assert rows[1].source_type == "text"
    assert document.status == "ready" and document.status_detail is None
    assert document.page_count == 5


def test_only_scanned_pages_are_sent_to_ocr(db_session, tmp_path, fast_embedding) -> None:
    path = _mixed_pdf(tmp_path / "mixed.pdf")
    engine = FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)])

    _ingest(db_session, path, engine)

    assert engine.calls == 1  # 5 pages: 3 native, 1 blank, 1 scanned -> exactly one OCR call


def test_a_document_with_no_scanned_pages_never_touches_the_ocr_engine(db_session, tmp_path, fast_embedding) -> None:
    path = build_pdf(tmp_path / "native.pdf", [add_text_page, add_logo_text_page])
    engine = FakeOcrEngine(RuntimeError("must not be called"))

    document, _ = _ingest(db_session, path, engine)

    assert engine.calls == 0
    assert document.status == "ready"


def test_ocr_page_keeps_its_real_pdf_page_number_in_every_record(db_session, tmp_path, fast_embedding) -> None:
    path = _mixed_pdf(tmp_path / "mixed.pdf")
    _ingest(db_session, path, FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)]))

    ocr_chunks = db_session.scalars(select(Chunk).where(Chunk.document_id == "doc-1", Chunk.source_type == "ocr")).all()

    assert ocr_chunks
    assert {chunk.page_number for chunk in ocr_chunks} == {3}  # PDF page 3 -- not 1, not 2, not 4
    assert all(chunk.id.startswith("doc-1-3-") for chunk in ocr_chunks)
    assert all(chunk.bbox is not None for chunk in ocr_chunks)
    # native chunks around it are untouched and keep source_type text
    native_pages = {chunk.page_number for chunk in db_session.scalars(select(Chunk).where(Chunk.source_type == "text")).all()}
    assert native_pages == {1, 4, 5}


def test_scanned_page_deep_in_a_document_is_not_off_by_one(db_session, tmp_path, fast_embedding) -> None:
    builders = [add_text_page] * 6 + [lambda doc: add_scanned_page(doc, SCANNED_TEXT)] + [add_text_page] * 3
    path = build_pdf(tmp_path / "long.pdf", builders)
    _ingest(db_session, path, FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)]))

    ocr_rows = [row for row in _page_rows(db_session).values() if row.status == "ocr"]
    assert [row.page_number for row in ocr_rows] == [7]
    assert {chunk.page_number for chunk in db_session.scalars(select(Chunk).where(Chunk.source_type == "ocr")).all()} == {7}


def test_ingestion_reports_each_stage_in_order(db_session, tmp_path, fast_embedding, monkeypatch) -> None:
    stages: list[str] = []
    original = documents_module._report

    def spy(db, document, detail):
        stages.append(detail)
        original(db, document, detail)

    monkeypatch.setattr(documents_module, "_report", spy)
    path = build_pdf(tmp_path / "two-scans.pdf", [add_text_page, lambda d: add_scanned_page(d, SCANNED_TEXT), lambda d: add_scanned_page(d, SCANNED_TEXT)])

    _ingest(db_session, path, FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)]))

    assert stages == [
        "Extracting text",
        "OCR required on 2 pages",
        "OCR processing page 2 (1 of 2)",
        "OCR processing page 3 (2 of 2)",
        "Indexing",
    ]


def test_native_only_documents_skip_the_ocr_stages(db_session, tmp_path, fast_embedding, monkeypatch) -> None:
    stages: list[str] = []
    original = documents_module._report
    monkeypatch.setattr(documents_module, "_report", lambda db, document, detail: (stages.append(detail), original(db, document, detail)))
    path = build_pdf(tmp_path / "native.pdf", [add_text_page])

    _ingest(db_session, path, FakeOcrEngine())

    assert stages == ["Extracting text", "Indexing"]


def test_partial_ocr_failure_is_reported_not_silently_treated_as_empty(db_session, tmp_path, fast_embedding) -> None:
    path = build_pdf(tmp_path / "partial.pdf", [add_text_page, lambda d: add_scanned_page(d, SCANNED_TEXT), add_text_page])

    document, _ = _ingest(db_session, path, FakeOcrEngine(RuntimeError("model crashed")))

    rows = _page_rows(db_session)
    assert rows[2].status == "failed" and "model crashed" in rows[2].status_detail
    assert document.status == "ready"  # the native pages are still usable
    assert "Some pages could not be processed with OCR" in document.status_detail
    assert "page 2" in document.status_detail


def test_all_scanned_document_without_ocr_fails_with_setup_guidance_not_empty(db_session, tmp_path, fast_embedding) -> None:
    path = build_pdf(tmp_path / "all-scanned.pdf", [lambda d: add_scanned_page(d, SCANNED_TEXT)] * 3)
    engine = FakeOcrEngine(available=False, reason=OCR_SETUP_MESSAGE)

    document, collection = _ingest(db_session, path, engine)

    assert document.status == "failed"  # NOT "empty": there IS content, it just couldn't be read
    assert "Some pages could not be processed with OCR" in document.status_detail
    assert "pip install rapidocr" in document.status_detail
    assert collection.added == []
    assert {row.status for row in _page_rows(db_session).values()} == {"failed"}
    assert engine.calls == 0


def test_genuinely_blank_document_is_empty_and_says_so(db_session, tmp_path, fast_embedding) -> None:
    path = build_pdf(tmp_path / "blank.pdf", [add_blank_page, add_blank_page])

    document, _ = _ingest(db_session, path, FakeOcrEngine())

    assert document.status == "empty"
    assert "blank" in document.status_detail.lower()
    assert {row.status for row in _page_rows(db_session).values()} == {"empty"}


def test_image_page_where_ocr_finds_no_text_is_empty_not_failed(db_session, tmp_path, fast_embedding) -> None:
    path = build_pdf(tmp_path / "photo.pdf", [lambda d: add_scanned_page(d, "")])

    document, _ = _ingest(db_session, path, FakeOcrEngine([]))

    assert _page_rows(db_session)[1].status == "empty"
    assert document.status == "empty"


def test_ocr_page_limit_skips_the_excess_with_a_clear_reason(db_session, tmp_path, fast_embedding, monkeypatch, settings) -> None:
    monkeypatch.setattr(settings, "ocr_max_pages", 1)
    path = build_pdf(tmp_path / "many-scans.pdf", [lambda d: add_scanned_page(d, SCANNED_TEXT)] * 3)
    engine = FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)])

    document, _ = _ingest(db_session, path, engine)

    rows = _page_rows(db_session)
    assert engine.calls == 1
    assert rows[1].status == "ocr"
    assert rows[2].status == rows[3].status == "failed"
    assert "limited to 1 scanned pages" in rows[2].status_detail
    assert document.status == "ready"
    assert "page 2, 3" in document.status_detail


# ---------- OCR text goes through the existing retrieval pipeline ----------


def test_ocr_text_is_embedded_and_keyword_indexed_like_native_text(db_session, tmp_path, fast_embedding) -> None:
    path = _mixed_pdf(tmp_path / "mixed.pdf")
    _, collection = _ingest(db_session, path, FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)]))

    embedded_documents = [text for batch in collection.added for text in batch["documents"]]
    assert any("MQTT broker handles telemetry" in text for text in embedded_documents)  # vector index
    metadata_pages = {meta["page_number"] for batch in collection.added for meta in batch["metadatas"]}
    assert 3 in metadata_pages

    hits = keyword_index.search(db_session, ["mqtt", "telemetry"], ["doc-1"], limit=5)  # FTS5
    hit_chunks = {chunk.id: chunk for chunk in db_session.scalars(select(Chunk).where(Chunk.id.in_([hit for hit, _ in hits]))).all()}
    assert hit_chunks and all(chunk.page_number == 3 and chunk.source_type == "ocr" for chunk in hit_chunks.values())


def test_page_text_search_cache_includes_ocr_text_and_marks_its_blocks(db_session, tmp_path, fast_embedding) -> None:
    from app.services.search import search_pages

    path = _mixed_pdf(tmp_path / "mixed.pdf")
    _ingest(db_session, path, FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)]))

    matches = search_pages(list(_page_rows(db_session).values()), "battery life")

    assert [match["page"] for match in matches] == [3]
    assert matches[0]["bbox"] is not None
    blocks = json.loads(_page_rows(db_session)[3].blocks_json)
    assert blocks[0]["source"] == "ocr"


def test_retrieval_finds_ocr_text_and_the_citation_points_at_the_real_pdf_page(db_session, tmp_path, fast_embedding) -> None:
    path = _mixed_pdf(tmp_path / "mixed.pdf")
    _ingest(db_session, path, FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)]))

    plan = RetrievalPlan(mode=RetrievalMode.FACT, query="How long does the battery last?", document_ids=["doc-1"], top_k=4)
    with use_providers(embedding=FakeEmbeddingProvider()):
        sources = asyncio.run(retrieve_with_plan(db_session, RecordingCollection(), plan))

    ocr_sources = [source for source in sources if source["source_type"] == "ocr"]
    assert ocr_sources and ocr_sources[0]["page_number"] == 3
    citations = build_citations(db_session, sources, [sources.index(ocr_sources[0]) + 1])
    assert citations[0]["page_number"] == 3
    assert citations[0]["source_type"] == "ocr"
    assert citations[0]["bbox_source"] == "exact" and citations[0]["bbox"]
    assert citations[0]["document_id"] == "doc-1"


def test_ocr_sources_are_flagged_to_the_model_as_possibly_noisy() -> None:
    block = format_source_block({"source_type": "ocr", "filename": "scan.pdf", "page_number": 3, "content": "text"}, 1)
    assert "page 3" in block and "OCR text" in block
    native = format_source_block({"source_type": "text", "filename": "a.pdf", "page_number": 3, "content": "text"}, 1)
    assert "OCR" not in native


def test_chat_answers_from_ocr_text_and_cites_the_ocr_page(db_session, tmp_path, fast_embedding, monkeypatch) -> None:
    path = _mixed_pdf(tmp_path / "mixed.pdf")
    _ingest(db_session, path, FakeOcrEngine([_ocr_line(LONG_OCR_TEXT)]))
    monkeypatch.setattr(chat_module, "get_collection", lambda: RecordingCollection())
    llm = FakeLLMProvider(lambda messages: json.dumps({"answer": "The sensor battery lasts 18 months. [1]", "source_ids": [1]}))

    with use_providers(llm=llm, embedding=FakeEmbeddingProvider()):
        result = asyncio.run(ask(ChatRequest(question="How long does the battery life last on the wireless sensor nodes?", document_ids=["doc-1"]), db=db_session))

    assert "18 months" in result["answer"]
    assert result["citations"][0]["page_number"] == 3
    assert result["citations"][0]["source_type"] == "ocr"
    assert "OCR text" in llm.calls[0]["messages"][-1]["content"]


def test_learning_modes_can_draw_on_ocr_text(db_session, tmp_path, fast_embedding) -> None:
    """Tutor / Exam / Brainstorm / Abstract build their evidence from these
    same chunk-backed samplers, so OCR chunks are eligible for all of them."""
    long_text = LONG_OCR_TEXT + " Gateways buffer readings locally whenever the uplink to the cloud is unavailable for a while."
    path = build_pdf(tmp_path / "scan-only.pdf", [lambda d: add_scanned_page(d, SCANNED_TEXT)])
    _ingest(db_session, path, FakeOcrEngine([_ocr_line(long_text)]))

    sampled = sample_document_sources(db_session, ["doc-1"], limit=5)
    overview = retrieve_overview(db_session, ["doc-1"], limit=5)

    assert sampled and sampled[0]["source_type"] == "ocr" and sampled[0]["page_number"] == 1
    assert overview and overview[0]["source_type"] == "ocr"


# ---------- the real local engine (skipped when not installed) ----------

real_engine_missing = not RapidOcrEngine().availability()[0]


@pytest.mark.skipif(real_engine_missing, reason="RapidOCR is not installed in this environment (pip install rapidocr)")
def test_real_engine_reads_a_scanned_page_end_to_end(db_session, tmp_path, fast_embedding) -> None:
    path = _mixed_pdf(tmp_path / "mixed.pdf")

    document = Document(id="real-1", filename="mixed.pdf", status="processing")
    db_session.add(document)
    db_session.commit()
    with use_providers(embedding=FakeEmbeddingProvider(), llm=FakeLLMProvider("[]")):
        asyncio.run(process_document(db_session, document, path, RecordingCollection(), provider=document_provider_for(path)))

    row = _page_rows(db_session, "real-1")[3]
    assert row.status == "ocr" and row.ocr_confidence and row.ocr_confidence > 0.8
    lowered = row.content.lower()
    assert "mqtt" in lowered and "battery life" in lowered and "18 months" in lowered
    assert document.status == "ready"
