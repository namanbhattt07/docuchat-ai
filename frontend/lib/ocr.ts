import type { DocumentItem, PageInfo, PageSummary } from "./api";

// Group 7 OCR / ingestion status: turns the backend's explicit document and
// per-page status into UI text. Pure functions, no React/fetch, so the wording
// rules -- especially "never present a failed OCR as an empty document" -- are
// unit-tested (tests/group7.test.mjs).

export type StatusTone = "ready" | "processing" | "warning" | "failed";
// `title` is the longer tooltip for the label (the page-by-page breakdown).
// `badge` is a short count shown beside the label ("2 OCR"); kept separate so the label never wraps mid-phrase.
export type DocumentStatusView = { label: string; badge: string | null; note: string | null; tone: StatusTone; noteTone: StatusTone | null; title: string | null };

// The backend words an OCR problem exactly this way (services/documents.py::_final_status);
// it is how an OCR failure is told apart from a generic ingestion failure.
export const OCR_FAILURE_PREFIX = "Some pages could not be processed with OCR";

export const isOcrFailure = (detail: string | null | undefined): boolean => Boolean(detail && detail.startsWith(OCR_FAILURE_PREFIX));

export function documentStatusView(doc: Pick<DocumentItem, "status" | "status_detail" | "page_count" | "page_summary">): DocumentStatusView {
  const detail = doc.status_detail?.trim() || null;
  if (doc.status === "processing") {
    // Live stage from ingestion: "Extracting text", "OCR required on 4 pages",
    // "OCR processing page 2 (1 of 4)", "Indexing".
    return { label: detail ?? "Processing…", badge: null, note: null, tone: "processing", noteTone: null, title: null };
  }
  if (doc.status === "ready") {
    const pages = `${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`;
    const ocrPages = doc.page_summary?.ocr ?? 0;
    return {
      label: `Ready · ${pages}`,
      badge: ocrPages > 0 ? `${ocrPages} OCR` : null,
      note: detail,
      tone: "ready",
      noteTone: detail ? "warning" : null,
      title: pageSummaryText(doc.page_summary, doc.page_count),
    };
  }
  if (doc.status === "empty") {
    return { label: "No readable text", badge: null, note: detail, tone: "warning", noteTone: "warning", title: null };
  }
  return { label: isOcrFailure(detail) ? "OCR failed" : "Failed", badge: null, note: detail, tone: "failed", noteTone: "failed", title: null };
}

export function pageSummaryText(summary: PageSummary | undefined, pageCount: number): string {
  if (!summary) return `${pageCount} page${pageCount === 1 ? "" : "s"}`;
  const parts = [`${pageCount} page${pageCount === 1 ? "" : "s"}`];
  if (summary.ocr) parts.push(`${summary.ocr} read by OCR`);
  if (summary.empty) parts.push(`${summary.empty} blank/no text`);
  if (summary.failed) parts.push(`${summary.failed} OCR failed`);
  return parts.join(" · ");
}

export type PageChip = { label: string; tone: StatusTone; title: string };

/** A chip for the page the viewer is on -- only for pages where the status is
 * worth surfacing (OCR'd, blank, OCR failed). Ordinary text pages get none. */
export function pageChip(page: Pick<PageInfo, "page_number" | "status" | "ocr_confidence" | "status_detail"> | undefined): PageChip | null {
  if (!page) return null;
  if (page.status === "ocr") {
    const confidence = page.ocr_confidence != null ? ` (confidence ${Math.round(page.ocr_confidence * 100)}%)` : "";
    return { label: `Page ${page.page_number} · OCR text`, tone: "warning", title: `This page is a scan; its text was recognized by OCR${confidence} and may contain errors.` };
  }
  if (page.status === "failed") {
    return { label: `Page ${page.page_number} · OCR failed`, tone: "failed", title: page.status_detail ?? "OCR could not read this page." };
  }
  if (page.status === "empty") {
    return { label: `Page ${page.page_number} · no text`, tone: "ready", title: page.status_detail ?? "No readable text on this page." };
  }
  return null;
}
