import asyncio
import json
from pathlib import Path

import fitz
import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider, FakeOcrEngine, FakeVisionProvider
from fastapi import HTTPException
from ingest import RecordingCollection, ingest
from pdfs import add_blank_page, add_figure_page, add_logo_text_page, add_scanned_page, add_text_page, build_pdf
from sqlalchemy import select

import app.api.v1.chat as chat_module
import app.api.v1.documents as documents_api
import app.services.documents as documents_module
from app.api.v1.chat import ChatRequest, ask
from app.core.config import get_settings
from app.models import Document, DocumentImage
from app.services import visual_qa, visuals
from app.services.ocr import OcrLine
from app.services.pdf_provider import PdfDocumentProvider, figure_captions
from app.services.document_model import DocumentBlock
from app.services.providers import ProviderUnavailable, set_vision_provider, use_providers

# ---------------------------------------------------------------------------
# Group 7 / 27: image + figure handling (limited scope). Detection, metadata,
# page/figure rendering, and -- above all -- that visual Q&A is honest about
# what this install can and cannot do.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fast_embedding(monkeypatch):
    async def fake_embed_in_batches(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", fake_embed_in_batches)


@pytest.fixture()
def processed_dir(tmp_path, monkeypatch):
    directory = tmp_path / "processed"
    monkeypatch.setattr(get_settings(), "processed_directory", str(directory))
    return directory


def _figure_pdf(path):
    """1 plain text | 2 figure page (image + 'Figure 1' caption) | 3 text | 4 logo page"""
    return build_pdf(path, [add_text_page, add_figure_page, add_text_page, add_logo_text_page])


# ---------- detection ----------


def test_scan_detects_the_figure_and_its_caption_but_not_decorations(tmp_path) -> None:
    scan = PdfDocumentProvider(ocr_engine=FakeOcrEngine()).scan(_figure_pdf(tmp_path / "f.pdf"))

    assert [(image.page_number, image.kind) for image in scan.images] == [(2, "image"), (2, "figure_caption")]
    raster, caption = scan.images
    assert raster.bbox == pytest.approx((100, 200, 400, 420), abs=1.0)
    assert (raster.width, raster.height) == (288, 211) or raster.width > 0  # source pixel size is recorded
    assert raster.xref  # reference back into the PDF
    assert caption.label == "Figure 1"
    assert caption.caption.startswith("Figure 1: Sensor throughput")
    # the small logo on page 4 is decorative, not a figure
    assert all(image.page_number != 4 for image in scan.images)


def test_text_only_document_has_no_figures(tmp_path) -> None:
    scan = PdfDocumentProvider(ocr_engine=FakeOcrEngine()).scan(build_pdf(tmp_path / "t.pdf", [add_text_page, add_text_page]))
    assert scan.images == []


def test_a_scanned_page_is_not_reported_as_a_figure_of_itself(tmp_path) -> None:
    path = build_pdf(tmp_path / "s.pdf", [lambda doc: add_scanned_page(doc, "Some scanned text about sensors and gateways.")])
    provider = PdfDocumentProvider(ocr_engine=FakeOcrEngine([OcrLine("Some scanned text about sensors and gateways.", (100, 100, 900, 160), 0.95)]))
    page = provider.scan(path).pages[0]

    resolved = provider.recognize_page(path, page)

    assert resolved.status == "ocr"
    assert [image for image in resolved.images if image.kind == "image"] == []  # the scan itself is not a figure


def test_a_full_page_photo_with_no_text_stays_a_visual_element(tmp_path) -> None:
    path = build_pdf(tmp_path / "photo.pdf", [lambda doc: add_scanned_page(doc, "")])
    provider = PdfDocumentProvider(ocr_engine=FakeOcrEngine([]))
    resolved = provider.recognize_page(path, provider.scan(path).pages[0])

    assert resolved.status == "empty"
    assert [image.kind for image in resolved.images] == ["image"]


def test_caption_detection_requires_a_separator_so_prose_is_not_a_caption() -> None:
    def block(text):
        return DocumentBlock(text, (0, 0, 1, 1), 0, len(text), False)

    found = figure_captions(4, [
        block("Figure 3: Throughput by protocol."),
        block("Fig. 4 - Gateway architecture"),
        block("Figure 5 shows the results of the second experiment."),  # prose, not a caption
        block("As seen in Figure 2: the trend is clear."),               # mid-sentence
        block("Figure 3: A duplicate label on the same page."),
    ], start_index=0)

    assert [image.label for image in found] == ["Figure 3", "Figure 4"]
    assert [image.image_index for image in found] == [0, 1]


# ---------- metadata persistence ----------


def test_ingestion_stores_figure_metadata_as_references_not_pixels(db_session, tmp_path) -> None:
    document, _ = ingest(db_session, _figure_pdf(tmp_path / "f.pdf"))

    rows = db_session.scalars(select(DocumentImage).where(DocumentImage.document_id == "doc-1").order_by(DocumentImage.kind)).all()

    assert document.visuals_indexed == 1
    assert [(row.page_number, row.kind, row.label) for row in rows] == [(2, "figure_caption", "Figure 1"), (2, "image", None)]
    image = rows[1]
    assert image.document_id == "doc-1" and image.xref and image.width and image.height
    assert image.bbox == pytest.approx([100, 200, 400, 420], abs=1.0)
    # metadata only: every stored value is a small scalar/JSON box, never bytes
    assert not any(isinstance(getattr(image, column.name), (bytes, bytearray)) for column in DocumentImage.__table__.columns)
    assert not any(column.type.__class__.__name__ in ("LargeBinary", "BLOB") for column in DocumentImage.__table__.columns)


def test_ocr_of_a_scanned_page_can_add_a_figure_caption(db_session, tmp_path) -> None:
    path = build_pdf(tmp_path / "s.pdf", [lambda doc: add_scanned_page(doc, "x")])
    engine = FakeOcrEngine([OcrLine("Figure 7: Signal strength over distance.", (100, 300, 1000, 350), 0.96)])

    ingest(db_session, path, engine)

    rows = db_session.scalars(select(DocumentImage)).all()
    assert [(row.kind, row.label, row.page_number) for row in rows] == [("figure_caption", "Figure 7", 1)]


def test_legacy_document_gets_figure_metadata_lazily_without_touching_chunks(db_session, tmp_path, monkeypatch) -> None:
    upload = tmp_path / "uploads"
    upload.mkdir()
    _figure_pdf(upload / "legacy.pdf")
    document = Document(id="legacy", filename="legacy.pdf", status="ready", page_count=4)  # visuals_indexed defaults to 0
    db_session.add(document)
    db_session.commit()
    monkeypatch.setattr(get_settings(), "upload_directory", str(upload))

    response = documents_api.get_document_images("legacy", page=None, db=db_session)

    assert document.visuals_indexed == 1
    assert response["total"] == 2
    assert {image["kind"] for image in response["images"]} == {"image", "figure_caption"}
    # a second call reads the stored rows (no re-scan needed)
    (upload / "legacy.pdf").unlink()
    assert documents_api.get_document_images("legacy", page=None, db=db_session)["total"] == 2


# ---------- page rendering ----------


def test_render_page_returns_a_png_of_the_requested_size(tmp_path) -> None:
    path = build_pdf(tmp_path / "p.pdf", [add_text_page])
    provider = PdfDocumentProvider()

    at_72 = fitz.Pixmap(provider.render_page(path, 1, dpi=72))
    at_144 = fitz.Pixmap(provider.render_page(path, 1, dpi=144))

    assert (at_72.width, at_72.height) == (595, 842)
    assert (at_144.width, at_144.height) == (1190, 1684)


def test_render_page_rejects_out_of_range_pages(tmp_path) -> None:
    path = build_pdf(tmp_path / "p.pdf", [add_text_page])
    with pytest.raises(IndexError):
        PdfDocumentProvider().render_page(path, 2, dpi=72)
    with pytest.raises(IndexError):
        PdfDocumentProvider().render_page(path, 0, dpi=72)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_render_clip_uses_native_coordinates_even_on_rotated_pages(tmp_path, rotation) -> None:
    doc = fitz.open()
    source = fitz.open()
    source_page = source.new_page(width=400, height=600)
    source_page.draw_rect(fitz.Rect(60, 110, 260, 150), color=None, fill=(0, 0, 0))
    page = doc.new_page(width=400, height=600)
    page.insert_image(page.rect, stream=source_page.get_pixmap(dpi=72).tobytes("png"))
    page.set_rotation(rotation)
    path = tmp_path / "r.pdf"
    doc.save(path)

    clip = fitz.Pixmap(PdfDocumentProvider().render_page(path, 1, dpi=72, clip=(60, 110, 260, 150)))

    samples = memoryview(clip.samples)
    dark = sum(1 for index in range(0, len(samples), clip.n) if samples[index] < 128)
    assert dark / (clip.width * clip.height) > 0.95  # the clip captured the marker, not blank paper


def test_page_image_is_cached_on_disk_and_reused(tmp_path, processed_dir) -> None:
    path = build_pdf(tmp_path / "p.pdf", [add_text_page])
    calls: list[int] = []

    class CountingProvider(PdfDocumentProvider):
        def render_page(self, path, page_number, *, dpi, clip=None):
            calls.append(dpi)
            return super().render_page(path, page_number, dpi=dpi, clip=clip)

    first = visuals.render_page_image("doc-1", path, 1, 100, provider=CountingProvider())
    second = visuals.render_page_image("doc-1", path, 1, 100, provider=CountingProvider())

    assert first == second and first.exists()
    assert processed_dir in first.parents
    assert calls == [100]  # rendered once


def test_page_render_dpi_is_clamped(tmp_path, processed_dir) -> None:
    path = build_pdf(tmp_path / "p.pdf", [add_text_page])
    huge = visuals.render_page_image("doc-1", path, 1, 5000)
    tiny = visuals.render_page_image("doc-1", path, 1, 1)

    assert huge.name == "1@200.png" and fitz.Pixmap(str(huge)).width == round(595 * 200 / 72)
    assert tiny.name == "1@50.png"


def test_figure_preview_is_a_padded_crop_of_the_figure(db_session, tmp_path, processed_dir) -> None:
    path = _figure_pdf(tmp_path / "f.pdf")
    ingest(db_session, path)
    image = db_session.scalar(select(DocumentImage).where(DocumentImage.kind == "image"))

    preview = visuals.render_figure_preview("doc-1", path, image)

    pixmap = fitz.Pixmap(str(preview))
    expected_width = round((300 + 2 * visuals.FIGURE_PREVIEW_PAD_PT) * visuals.FIGURE_PREVIEW_DPI / 72)
    assert abs(pixmap.width - expected_width) <= 2
    assert pixmap.width < 595 * visuals.FIGURE_PREVIEW_DPI / 72  # a crop, not the whole page


def test_caption_only_figure_previews_fall_back_to_the_page(db_session, tmp_path, processed_dir) -> None:
    path = _figure_pdf(tmp_path / "f.pdf")
    ingest(db_session, path)
    caption = db_session.scalar(select(DocumentImage).where(DocumentImage.kind == "figure_caption"))

    preview = visuals.render_figure_preview("doc-1", path, caption)

    assert "pages" in preview.parts and preview.name.startswith("2@")


# ---------- API endpoints ----------


@pytest.fixture()
def api_env(db_session, tmp_path, monkeypatch, processed_dir):
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(get_settings(), "upload_directory", str(upload))
    return upload


def _api_document(db_session, upload: Path, document_id="doc-1"):
    path = _figure_pdf(upload / f"{document_id}.pdf")
    ingest(db_session, path, document_id=document_id)
    document = db_session.get(Document, document_id)
    return document


def test_pages_endpoint_reports_status_and_figure_counts(db_session, api_env) -> None:
    path = build_pdf(api_env / "doc-1.pdf", [add_text_page, add_blank_page, add_figure_page, lambda d: add_scanned_page(d, "Battery life is 18 months for the sensor nodes.")])
    ingest(db_session, path, FakeOcrEngine([OcrLine("Battery life is 18 months for the sensor nodes.", (100, 100, 900, 160), 0.9)]))

    pages = documents_api.get_document_pages("doc-1", db=db_session)["pages"]

    assert [(p["page_number"], p["status"], p["figure_count"]) for p in pages] == [(1, "text", 0), (2, "empty", 0), (3, "text", 2), (4, "ocr", 0)]
    assert pages[3]["source_type"] == "ocr" and pages[3]["ocr_confidence"] == 0.9
    assert pages[1]["status_detail"] == "Blank page."


def test_documents_list_includes_a_page_status_summary(db_session, api_env) -> None:
    path = build_pdf(api_env / "doc-1.pdf", [add_text_page, add_blank_page, lambda d: add_scanned_page(d, "Sensor gateway firmware notes for the whole installation.")])
    ingest(db_session, path, FakeOcrEngine([OcrLine("Sensor gateway firmware notes for the whole installation.", (100, 100, 900, 160), 0.9)]))
    db_session.add(Document(id="legacy", filename="old.pdf", status="ready"))
    db_session.commit()

    listing = {item["id"]: item for item in documents_api.list_documents(db=db_session)}

    assert listing["doc-1"]["page_summary"] == {"text": 1, "ocr": 1, "empty": 1, "failed": 0, "unknown": 0}
    assert listing["legacy"]["page_summary"]["unknown"] == 0  # no pages at all -> all zeros


def test_page_image_endpoint_serves_png_and_rejects_bad_pages(db_session, api_env) -> None:
    document = _api_document(db_session, api_env)

    response = documents_api.get_page_image(document.id, 2, dpi=72, db=db_session)

    assert response.media_type == "image/png"
    assert fitz.Pixmap(str(response.path)).width == 595
    for bad_page in (0, 5, 99):
        with pytest.raises(HTTPException) as exc_info:
            documents_api.get_page_image(document.id, bad_page, dpi=None, db=db_session)
        assert exc_info.value.status_code == 404
    with pytest.raises(HTTPException) as exc_info:
        documents_api.get_page_image("missing", 1, dpi=None, db=db_session)
    assert exc_info.value.status_code == 404


def test_images_endpoint_lists_figures_with_preview_urls_and_filters_by_page(db_session, api_env) -> None:
    document = _api_document(db_session, api_env)

    everything = documents_api.get_document_images(document.id, page=None, db=db_session)
    on_page_one = documents_api.get_document_images(document.id, page=1, db=db_session)

    assert everything["total"] == 2 and on_page_one["total"] == 0
    raster = next(image for image in everything["images"] if image["kind"] == "image")
    assert raster["page_number"] == 2 and raster["bbox"] and raster["width"] and raster["xref"]
    assert raster["preview_url"] == f"/api/v1/documents/doc-1/images/{raster['id']}/preview"
    assert raster["page_image_url"] == "/api/v1/documents/doc-1/pages/2/image"


def test_image_preview_endpoint_is_scoped_to_its_own_document(db_session, api_env) -> None:
    first = _api_document(db_session, api_env, "doc-1")
    _api_document(db_session, api_env, "doc-2")
    image_id = documents_api.get_document_images(first.id, page=None, db=db_session)["images"][0]["id"]

    assert documents_api.get_image_preview("doc-1", image_id, db=db_session).media_type == "image/png"
    with pytest.raises(HTTPException) as exc_info:  # doc-2 must not serve doc-1's figure
        documents_api.get_image_preview("doc-2", image_id, db=db_session)
    assert exc_info.value.status_code == 404


def test_deleting_a_document_removes_its_figures_and_render_cache(db_session, api_env, processed_dir, monkeypatch) -> None:
    class Collection:
        def delete(self, where):
            pass

    monkeypatch.setattr(documents_api, "get_collection", lambda: Collection())
    document = _api_document(db_session, api_env)
    documents_api.get_page_image(document.id, 2, dpi=72, db=db_session)
    assert (processed_dir / "doc-1").exists()

    documents_api.delete_document(document.id, db=db_session)

    assert db_session.scalars(select(DocumentImage)).all() == []
    assert not (processed_dir / "doc-1").exists()


# ---------- visual question detection ----------


@pytest.mark.parametrize("question", [
    "What does Figure 1 show?",
    "Describe the diagram on page 2",
    "What is the trend in the chart?",
    "Explain the graph of sensor throughput",
    "What's in the picture on page 5?",
    "Summarize the image",
])
def test_visual_questions_are_recognized(question) -> None:
    assert visual_qa.looks_visual(question)


@pytest.mark.parametrize("question", [
    "What is a graph database?",                     # noun, no intent
    "What does the document say about image compression?",
    "List the captions of all figures",              # explicitly text
    "Summarize the methodology section",
    "Can you figure out the trend in battery usage?",  # "figure" the verb
    "How does ZigBee routing work?",
    "What is the battery life?",
])
def test_ordinary_questions_are_not_visual(question) -> None:
    assert not visual_qa.looks_visual(question)


def test_explicit_label_and_page_extraction() -> None:
    assert visual_qa.explicit_label("what does fig. 3 show") == "Figure 3"
    assert visual_qa.explicit_label("describe Chart 2.1") == "Chart 2.1"
    assert visual_qa.explicit_label("what is shown") is None
    assert visual_qa.explicit_page("look at page 12") == 12


# ---------- unsupported visual Q&A through /chat ----------


class NoVectorHits(RecordingCollection):
    pass


@pytest.fixture()
def figure_chat_db(db_session, tmp_path, monkeypatch, processed_dir):
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(get_settings(), "upload_directory", str(upload))
    monkeypatch.setattr(get_settings(), "ollama_vision_model", "")
    monkeypatch.setattr(chat_module, "get_collection", lambda: NoVectorHits())
    path = build_pdf(upload / "doc-1.pdf", [add_text_page, add_figure_page, add_text_page])
    ingest(db_session, path, filename="protocols.pdf")
    return db_session


def _ask(db, question, llm=None, **kwargs):
    llm = llm or FakeLLMProvider(json.dumps({"answer": "A normal grounded answer. [1]", "source_ids": [1]}))
    with use_providers(llm=llm, embedding=FakeEmbeddingProvider()):
        result = asyncio.run(ask(ChatRequest(question=question, document_ids=["doc-1"], **kwargs), db=db))
    return result, llm


def test_visual_question_without_a_vision_model_is_answered_honestly(figure_chat_db) -> None:
    result, llm = _ask(figure_chat_db, "What does Figure 1 show?")

    assert "This document contains" in result["answer"]
    assert "on page 2" in result["answer"]
    assert "visual question answering is not supported by the current model configuration" in result["answer"]
    assert llm.calls == []  # the text model is never asked to "interpret" an image
    assert result["visual"]["supported"] is False
    assert "vision model" in result["visual"]["reason"].lower()


def test_unsupported_visual_response_never_claims_to_have_seen_the_chart(figure_chat_db) -> None:
    result, _ = _ask(figure_chat_db, "What does Figure 1 show?")
    lowered = result["answer"].lower()

    for forbidden in ("the chart shows", "the figure shows", "the graph shows", "the diagram shows", "bars", "increases", "trend"):
        assert forbidden not in lowered
    assert "not an interpretation of the image" in lowered  # extracted text is labelled as text


def test_unsupported_visual_response_still_gives_citations_previews_and_extracted_text(figure_chat_db) -> None:
    result, _ = _ask(figure_chat_db, "What does Figure 1 show?")

    citations = result["citations"]
    assert citations and all(citation["page_number"] == 2 for citation in citations)
    assert all(citation["document_id"] == "doc-1" and citation["filename"] == "protocols.pdf" for citation in citations)
    figure_citation = citations[0]
    assert figure_citation["source_type"] == "figure"
    assert figure_citation["bbox"] and figure_citation["bbox_source"] == "exact"  # navigation lands on the figure's box
    assert "Figure 1: Sensor throughput by protocol." in figure_citation["excerpt"]
    assert any("comparison of protocols".lower() in c["excerpt"].lower() or "chart below" in c["excerpt"].lower() for c in citations[1:])
    assert "[1]" in result["answer"] and "[2]" in result["answer"]

    target = result["visual"]["targets"][0]
    assert target["page_number"] == 2
    assert target["page_image_url"] == "/api/v1/documents/doc-1/pages/2/image"
    assert {figure["kind"] for figure in target["figures"]} == {"image", "figure_caption"}
    assert all(figure["preview_url"].startswith("/api/v1/documents/doc-1/images/") for figure in target["figures"])


def test_visual_question_by_page_number_targets_that_page(figure_chat_db) -> None:
    result, llm = _ask(figure_chat_db, "Describe the image on page 2")
    assert result["visual"]["targets"][0]["page_number"] == 2 and llm.calls == []


def test_visual_question_about_a_page_with_no_figure_falls_through_to_the_normal_answer(figure_chat_db) -> None:
    result, llm = _ask(figure_chat_db, "Describe the image on page 3")

    assert "visual" not in result
    assert "normal grounded answer" in result["answer"]
    assert len(llm.calls) == 1


def test_a_named_figure_that_does_not_exist_falls_through_rather_than_inventing_one(figure_chat_db) -> None:
    result, llm = _ask(figure_chat_db, "What does Figure 9 show?")
    assert "visual" not in result and len(llm.calls) == 1


def test_ordinary_questions_on_a_document_with_figures_use_the_normal_pipeline(figure_chat_db) -> None:
    result, llm = _ask(figure_chat_db, "What does the machine learning introduction say about supervised learning?")

    assert "visual" not in result
    assert "normal grounded answer" in result["answer"]
    assert len(llm.calls) == 1


def test_selection_questions_are_never_diverted_to_the_visual_path(figure_chat_db) -> None:
    result, _ = _ask(figure_chat_db, "Explain the figure text", selected_text="Figure 1: Sensor throughput by protocol.", selection_page=2, selection_action="explain")
    assert "visual" not in result


def test_visual_question_on_a_document_without_any_figures_is_answered_normally(db_session, tmp_path, monkeypatch, processed_dir) -> None:
    monkeypatch.setattr(chat_module, "get_collection", lambda: NoVectorHits())
    ingest(db_session, build_pdf(tmp_path / "plain.pdf", [add_text_page]))

    result, llm = _ask(db_session, "What does the chart show about machine learning?")

    assert "visual" not in result and len(llm.calls) == 1


def test_the_visual_turn_is_persisted_like_any_other_message(figure_chat_db) -> None:
    from app.models import Message

    result, _ = _ask(figure_chat_db, "What does Figure 1 show?")

    messages = figure_chat_db.scalars(select(Message).where(Message.conversation_id == result["conversation_id"]).order_by(Message.created_at)).all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert json.loads(messages[1].citations_json)[0]["page_number"] == 2
    assert result["message_id"] == messages[1].id


# ---------- a vision model, when (and only when) one really exists ----------


def test_with_a_vision_provider_the_page_image_is_sent_and_the_answer_is_labelled(figure_chat_db) -> None:
    vision = FakeVisionProvider("Three bars, the tallest labelled ZigBee.")
    set_vision_provider(vision)

    result, llm = _ask(figure_chat_db, "What does Figure 1 show?")

    assert llm.calls == []  # still not the text model
    assert len(vision.calls) == 1
    assert vision.calls[0]["image_bytes"] > 1000 and "Figure 1" in vision.calls[0]["question"]
    assert "Sensor" in vision.calls[0]["context"] or "chart below" in vision.calls[0]["context"]
    assert "Vision model interpretation (fake-vision:1b)" in result["answer"]
    assert "may be inaccurate" in result["answer"]
    assert "Three bars, the tallest labelled ZigBee." in result["answer"]
    assert "not supported" not in result["answer"]
    assert result["visual"]["supported"] is True and result["visual"]["model"] == "fake-vision:1b"
    assert result["citations"][0]["page_number"] == 2  # still cited to the page/figure


def test_a_failing_vision_provider_degrades_to_the_honest_response(figure_chat_db) -> None:
    class Failing(FakeVisionProvider):
        async def answer(self, image_png, question, context=""):
            raise ProviderUnavailable("vision model is not responding")

    set_vision_provider(Failing())

    result, _ = _ask(figure_chat_db, "What does Figure 1 show?")

    assert "not supported by the current model configuration" in result["answer"]
    assert "could not be used" in result["answer"] and "not responding" in result["answer"]
    assert result["visual"]["supported"] is False
    assert result["citations"]


def test_only_the_first_target_page_is_sent_to_the_vision_model(db_session, tmp_path, monkeypatch, processed_dir) -> None:
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(get_settings(), "upload_directory", str(upload))
    monkeypatch.setattr(chat_module, "get_collection", lambda: NoVectorHits())
    ingest(db_session, build_pdf(upload / "doc-1.pdf", [add_figure_page, add_figure_page]))
    vision = FakeVisionProvider()
    set_vision_provider(vision)

    result, _ = _ask(db_session, "Describe the chart that compares protocols and throughput")

    assert len(vision.calls) == 1
    assert {target["page_number"] for target in result["visual"]["targets"]} == {1, 2}


def test_system_status_reports_ocr_and_vision_honestly(monkeypatch) -> None:
    from app.api.v1.system import system_status
    from app.services import ocr as ocr_module
    from app.services.ocr import set_ocr_engine

    async def offline():
        return {"available": False, "generation_model_ready": False, "embedding_model_ready": False}

    import app.api.v1.system as system_module

    monkeypatch.setattr(system_module, "status", offline)
    monkeypatch.setattr(get_settings(), "ollama_vision_model", "")
    set_ocr_engine(FakeOcrEngine(available=False, reason=ocr_module.OCR_SETUP_MESSAGE))

    payload = asyncio.run(system_status())

    assert payload["ollama"]["available"] is False  # unchanged shape
    assert payload["ocr"]["available"] is False and "pip install rapidocr" in payload["ocr"]["detail"]
    assert payload["vision"]["supported"] is False and payload["vision"]["model"] is None
