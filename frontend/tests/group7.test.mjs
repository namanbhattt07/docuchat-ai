// Group 7 UI logic: OCR/ingestion status wording, prompt-template requests and
// validation, visual Q&A honesty (lib/ocr.ts, lib/templates.ts, lib/visual.ts).
// Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { documentStatusView, isOcrFailure, pageChip, pageSummaryText } from "../lib/ocr.ts";
import { buildTemplateRequest, draftFromTemplate, EMPTY_DRAFT, TEMPLATE_LIMITS, templateSections, validateTemplateDraft } from "../lib/templates.ts";
import { figureTitle, targetAsCitation, visualBanner } from "../lib/visual.ts";

const summary = (overrides = {}) => ({ text: 0, ocr: 0, empty: 0, failed: 0, unknown: 0, ...overrides });

// ---------- ingestion / OCR status ----------

test("a processing document shows the live ingestion stage", () => {
  const stages = ["Extracting text", "OCR required on 4 pages", "OCR processing page 3 (2 of 4)", "Indexing"];
  for (const stage of stages) {
    const view = documentStatusView({ status: "processing", status_detail: stage, page_count: 0 });
    assert.equal(view.label, stage);
    assert.equal(view.tone, "processing");
  }
  assert.equal(documentStatusView({ status: "processing", status_detail: null, page_count: 0 }).label, "Processing…");
});

test("a ready document reports how many pages came from OCR", () => {
  const plain = documentStatusView({ status: "ready", status_detail: null, page_count: 12, page_summary: summary({ text: 12 }) });
  assert.equal(plain.label, "Ready · 12 pages");
  assert.equal(plain.note, null);
  const scanned = documentStatusView({ status: "ready", status_detail: null, page_count: 12, page_summary: summary({ text: 8, ocr: 4 }) });
  assert.equal(scanned.label, "Ready · 12 pages");
  assert.equal(scanned.badge, "4 OCR");
  assert.equal(scanned.title, "12 pages · 4 read by OCR");
  assert.equal(plain.badge, null);
  assert.equal(documentStatusView({ status: "ready", status_detail: null, page_count: 1 }).label, "Ready · 1 page");
});

test("a ready document with failed OCR pages surfaces the warning instead of hiding it", () => {
  const detail = "Some pages could not be processed with OCR (page 3, 7). OCR is not available.";
  const view = documentStatusView({ status: "ready", status_detail: detail, page_count: 9, page_summary: summary({ text: 7, failed: 2 }) });
  assert.equal(view.tone, "ready");
  assert.equal(view.note, detail);
  assert.equal(view.noteTone, "warning");
});

test("failed OCR is labelled as OCR failure, distinct from an ingestion failure and from an empty document", () => {
  const ocr = documentStatusView({ status: "failed", status_detail: "Some pages could not be processed with OCR (page 1). Install rapidocr.", page_count: 3 });
  assert.equal(ocr.label, "OCR failed");
  assert.match(ocr.note, /rapidocr/);
  const ingestion = documentStatusView({ status: "failed", status_detail: "This PDF could not be processed.", page_count: 0 });
  assert.equal(ingestion.label, "Failed");
  const empty = documentStatusView({ status: "empty", status_detail: "No readable text was found: every page is blank.", page_count: 2 });
  assert.equal(empty.label, "No readable text");
  assert.equal(empty.tone, "warning");
  assert.ok(new Set([ocr.label, ingestion.label, empty.label]).size === 3);
});

test("isOcrFailure only matches the backend's OCR failure wording", () => {
  assert.equal(isOcrFailure("Some pages could not be processed with OCR (page 2)."), true);
  assert.equal(isOcrFailure("This PDF could not be processed."), false);
  assert.equal(isOcrFailure(null), false);
});

test("page summary text names each kind of page", () => {
  assert.equal(pageSummaryText(summary({ text: 6, ocr: 3, empty: 1, failed: 2 }), 12), "12 pages · 3 read by OCR · 1 blank/no text · 2 OCR failed");
  assert.equal(pageSummaryText(summary({ text: 4 }), 4), "4 pages");
  assert.equal(pageSummaryText(undefined, 1), "1 page");
});

test("page chips flag OCR'd, failed and blank pages but not ordinary text pages", () => {
  assert.equal(pageChip({ page_number: 2, status: "text", ocr_confidence: null, status_detail: null }), null);
  assert.equal(pageChip(undefined), null);
  const ocr = pageChip({ page_number: 3, status: "ocr", ocr_confidence: 0.912, status_detail: null });
  assert.equal(ocr.label, "Page 3 · OCR text");
  assert.match(ocr.title, /confidence 91%/);
  assert.match(ocr.title, /may contain errors/);
  assert.equal(pageChip({ page_number: 4, status: "failed", ocr_confidence: null, status_detail: "OCR failed on this page: boom" }).tone, "failed");
  assert.equal(pageChip({ page_number: 5, status: "empty", ocr_confidence: null, status_detail: "Blank page." }).label, "Page 5 · no text");
});

// ---------- prompt templates ----------

test("template sections are read from the output format headings", () => {
  assert.deepEqual(templateSections({ output_format: "## Facts\n## Issue\n\n### Rule" }), ["Facts", "Issue", "Rule"]);
  assert.deepEqual(templateSections({ output_format: null }), []);
});

test("applying a template sends its id and a readable message", () => {
  const template = { id: "builtin:case-brief", name: "Case Brief" };
  assert.deepEqual(buildTemplateRequest(template, ""), { question: "Apply the “Case Brief” template", templateId: "builtin:case-brief" });
  assert.equal(buildTemplateRequest(template, "  focus on liability ").question, "Case Brief: focus on liability");
});

test("template drafts are validated like the backend does", () => {
  const good = { name: "Risk Register", description: "", instruction: "List risks.", output_format: "## Risks" };
  assert.equal(validateTemplateDraft(good), null);
  assert.match(validateTemplateDraft({ ...good, name: "  " }), /name/i);
  assert.match(validateTemplateDraft({ ...good, instruction: " " }), /instruction/i);
  assert.match(validateTemplateDraft({ ...good, name: "n".repeat(TEMPLATE_LIMITS.name + 1) }), /80/);
  assert.match(validateTemplateDraft({ ...good, instruction: "x".repeat(TEMPLATE_LIMITS.instruction + 1) }), /2000/);
  assert.match(validateTemplateDraft({ ...good, description: "d".repeat(TEMPLATE_LIMITS.description + 1) }), /300/);
  assert.match(validateTemplateDraft({ ...good, output_format: "o".repeat(TEMPLATE_LIMITS.output_format + 1) }), /1000/);
  assert.match(validateTemplateDraft(EMPTY_DRAFT), /name/i);
});

test("editing a template starts from its saved values", () => {
  const draft = draftFromTemplate({ name: "N", description: "D", instruction: "I", output_format: null });
  assert.deepEqual(draft, { name: "N", description: "D", instruction: "I", output_format: "" });
});

// ---------- visual Q&A honesty ----------

test("without a vision model the banner says visual Q&A is unsupported and never claims an interpretation", () => {
  const banner = visualBanner({ supported: false, model: null, reason: "No local vision model is configured.", targets: [] });
  assert.equal(banner.tone, "unsupported");
  assert.match(banner.title, /aren't supported/);
  assert.equal(banner.detail, "No local vision model is configured.");
  assert.doesNotMatch(banner.title + banner.detail, /interpreted/i);
});

test("the interpreted banner needs both supported=true and a named model", () => {
  const withModel = visualBanner({ supported: true, model: "llava:7b", reason: "", targets: [] });
  assert.equal(withModel.tone, "interpreted");
  assert.match(withModel.title, /llava:7b/);
  assert.match(withModel.detail, /may be inaccurate/);
  // supported without a model name must not be presented as an interpretation
  assert.equal(visualBanner({ supported: true, model: null, reason: "x", targets: [] }).tone, "unsupported");
});

test("a preview can be opened in the viewer through a citation-shaped handle", () => {
  const target = { document_id: "doc-1", filename: "a.pdf", page_number: 7, page_image_url: "/x", figures: [] };
  const withBox = targetAsCitation(target, [1, 2, 3, 4]);
  assert.equal(withBox.page_number, 7);
  assert.equal(withBox.document_id, "doc-1");
  assert.deepEqual(withBox.bbox, [1, 2, 3, 4]);
  assert.equal(withBox.bbox_source, "exact");
  assert.equal(targetAsCitation(target).bbox_source, "unavailable");
});

test("figure titles prefer the detected label", () => {
  assert.equal(figureTitle({ label: "Figure 3", kind: "figure_caption" }, 5), "Figure 3");
  assert.equal(figureTitle({ label: null, kind: "image" }, 5), "Image on page 5");
  assert.equal(figureTitle({ label: null, kind: "figure_caption" }, 5), "Figure on page 5");
});
