import type { DocumentItem } from "./api";

// Group 8 document lifecycle, as the UI needs it: which documents can be chatted
// with, how an upload that didn't go through is worded, and when to keep asking
// for something that isn't ready yet. Pure functions (no React, no fetch) so the
// rules are unit-tested (tests/group8.test.mjs). Document *status wording* stays
// in lib/ocr.ts::documentStatusView -- this file adds no second status model.

type Statused = Pick<DocumentItem, "status">;

// The HTTP status carried by an ApiError (lib/api.ts) -- 0 when the backend
// could not be reached. Read structurally rather than with `instanceof`, like
// the other lib modules this one only imports *types* from ./api, which keeps
// it loadable by the Node test runner without a module graph.
function statusOf(error: unknown): number | undefined {
  const status = error instanceof Error ? (error as Error & { status?: unknown }).status : undefined;
  return typeof status === "number" ? status : undefined;
}

export const isChattable = (doc: Statused): boolean => doc.status === "ready";

/** Why a document can't be selected as chat context yet, or null when it can. */
export function selectionBlockedReason(doc: Statused): string | null {
  switch (doc.status) {
    case "ready": return null;
    case "processing": return "Still processing — you can chat with it once it shows Ready.";
    case "empty": return "This document has no readable text to chat with.";
    default: return "This document failed to process. Remove it and upload it again.";
  }
}

/** Keep only selected ids that still exist AND can be chatted with. A document
 * that is deleted, or that (re)enters a non-ready state, must not stay selected:
 * the backend would answer "nothing to search" for it while the UI still
 * claimed "1 source selected". */
export function pruneSelection(selected: string[], documents: DocumentItem[]): string[] {
  const usable = new Set(documents.filter(isChattable).map(doc => doc.id));
  const kept = selected.filter(id => usable.has(id));
  return kept.length === selected.length ? selected : kept;
}

export type FailureReport = { unreachable: true } | { unreachable: false; message: string };

/** How a failed user action is reported. "The backend can't be reached" is
 * shared state (the status card, the slow retry poll, a message that clears
 * itself when it comes back), not a one-off error string nothing would ever
 * clear -- so it is reported as that. Anything else is the error's own words. */
export function failureReport(error: unknown, fallback: string): FailureReport {
  if (statusOf(error) === 0) return { unreachable: true };
  return { unreachable: false, message: error instanceof Error && error.message ? error.message : fallback };
}

/** A jump (from a citation, the contents list or a figure) is aimed at one
 * document's pages. When a *different* document becomes the active one, its
 * viewer mounts fresh and must not inherit the previous document's jump --
 * "page 6" of a 3-page file is a "Failed to load the page" viewer. Staying on
 * the same document keeps the jump (and its highlight) alive. */
export function jumpForDocument<T>(jump: T | null, currentDocumentId: string | null, nextDocumentId: string | null): T | null {
  return currentDocumentId === nextDocumentId ? jump : null;
}

export type UploadNotice = { tone: "error" | "notice"; text: string };

/** What to tell the person when an upload did not go through. A duplicate is
 * not an error (the file is already there), so it gets the calmer tone; the
 * backend's own wording (which names the existing document, the size limit, or
 * why the file isn't a PDF) is kept as-is. */
export function uploadNotice(error: unknown): UploadNotice {
  const status = statusOf(error);
  const text = error instanceof Error && error.message ? error.message : "The upload failed. Please try again.";
  return { tone: status === 409 ? "notice" : "error", text };
}

export const SUGGESTION_POLL_MS = 3000;
export const SUGGESTION_POLL_LIMIT = 10;

/** Delay before asking for suggested questions again, or null to stop. They are
 * written a few seconds after a document turns Ready, so the first request can
 * legitimately answer "pending". Bounded, so a generation that never finishes
 * (model down) doesn't poll forever. */
export function nextSuggestionPoll(attempt: number, pending: boolean | undefined): number | null {
  return pending && attempt < SUGGESTION_POLL_LIMIT ? SUGGESTION_POLL_MS : null;
}

export const REFRESH_PROCESSING_MS = 2000;
export const REFRESH_UNREACHABLE_MS = 5000;

/** How often the library should be re-fetched, or null when nothing needs it. A
 * document still processing needs a fast poll; an unreachable backend needs a
 * slower retry (so a restarted backend is picked up without a manual reload). */
export function refreshInterval(documents: Statused[], unreachable: boolean): number | null {
  if (unreachable) return REFRESH_UNREACHABLE_MS;
  return documents.some(doc => doc.status === "processing") ? REFRESH_PROCESSING_MS : null;
}

/** Whether the "Model setup needed" card should be shown, and what is missing. */
export type SystemView = { ready: boolean; headline: string; detail: string };
export function systemView(system: {
  ollama: { available: boolean; generation_model_ready: boolean; embedding_model_ready: boolean; generation_model?: string; embedding_model?: string };
} | null, unreachable: boolean): SystemView {
  if (unreachable) return { ready: false, headline: "Backend not reachable", detail: "Start the DocuChat backend to continue" };
  if (!system) return { ready: false, headline: "Checking…", detail: "Looking for the local AI models" };
  const { ollama } = system;
  if (!ollama.available) return { ready: false, headline: "Ollama isn't running", detail: "Start Ollama, then reload this page" };
  const missing = [
    !ollama.generation_model_ready ? ollama.generation_model ?? "the answer model" : null,
    !ollama.embedding_model_ready ? ollama.embedding_model ?? "the embedding model" : null,
  ].filter((name): name is string => Boolean(name));
  if (missing.length > 0) return { ready: false, headline: "Model setup needed", detail: `Not installed in Ollama: ${missing.join(", ")}` };
  return { ready: true, headline: "Private AI is ready", detail: "Running locally on this Mac" };
}
