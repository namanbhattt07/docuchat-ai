// Group 8 UI logic: how failures reach the person (lib/api.ts), and the document
// lifecycle rules (lib/documents.ts): who can be selected, how an upload that
// didn't go through is worded, when to keep polling. Run with `npm test`.
import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { ApiError, isConnectionError, listDocuments, UNREACHABLE_MESSAGE, uploadDocument } from "../lib/api.ts";
import {
  failureReport, isChattable, jumpForDocument, nextSuggestionPoll, pruneSelection, refreshInterval, selectionBlockedReason, SUGGESTION_POLL_LIMIT, SUGGESTION_POLL_MS,
  systemView, uploadNotice,
} from "../lib/documents.ts";

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

const respond = (status, body) => async () => new Response(body === undefined ? null : JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
const doc = (id, status, extra = {}) => ({ id, filename: `${id}.pdf`, page_count: 3, status, status_detail: null, created_at: "", ...extra });

// ---------- failures reach the person as something they can act on ----------

test("an unreachable backend says so instead of the browser's bare 'Failed to fetch'", async () => {
  globalThis.fetch = async () => { throw new TypeError("Failed to fetch"); };
  await assert.rejects(listDocuments(), error => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.status, 0);
    assert.equal(error.message, UNREACHABLE_MESSAGE);
    assert.doesNotMatch(error.message, /Failed to fetch/);
    assert.equal(isConnectionError(error), true);
    return true;
  });
});

test("the backend's own explanation is shown as-is, with its status", async () => {
  globalThis.fetch = respond(409, { detail: 'This PDF is already in your library as "report.pdf".' });
  await assert.rejects(uploadDocument(new File(["x"], "copy.pdf")), error => {
    assert.equal(error.status, 409);
    assert.equal(error.message, 'This PDF is already in your library as "report.pdf".');
    assert.equal(isConnectionError(error), false);
    return true;
  });
});

test("validation errors are listed, never '[object Object]'", async () => {
  globalThis.fetch = respond(422, { detail: [{ msg: "field required" }, { msg: "value is not a valid integer" }] });
  await assert.rejects(listDocuments(), /field required; value is not a valid integer/);
});

test("a server crash and a malformed error body each get a plain sentence", async () => {
  globalThis.fetch = async () => new Response("<html>Internal Server Error</html>", { status: 500 });
  await assert.rejects(listDocuments(), /unexpected error/);
  globalThis.fetch = respond(400, { detail: { nested: "object" } });
  await assert.rejects(listDocuments(), /Something went wrong/);
});

test("a successful response and an empty 204 both resolve", async () => {
  globalThis.fetch = respond(200, [{ id: "a" }]);
  assert.deepEqual(await listDocuments(), [{ id: "a" }]);
  globalThis.fetch = async () => new Response(null, { status: 204 });
  assert.equal(await listDocuments(), undefined);
});

// ---------- who can be chatted with ----------

test("only a Ready document can be chat context, and each other state says why", () => {
  assert.equal(isChattable(doc("a", "ready")), true);
  for (const status of ["processing", "empty", "failed"]) assert.equal(isChattable(doc("a", status)), false);
  assert.equal(selectionBlockedReason(doc("a", "ready")), null);
  assert.match(selectionBlockedReason(doc("a", "processing")), /Still processing/);
  assert.match(selectionBlockedReason(doc("a", "empty")), /no readable text/);
  assert.match(selectionBlockedReason(doc("a", "failed")), /failed to process/);
  assert.equal(new Set(["processing", "empty", "failed"].map(status => selectionBlockedReason(doc("a", status)))).size, 3);
});

test("a selection drops documents that were deleted or stopped being usable, keeping order", () => {
  const library = [doc("a", "ready"), doc("b", "failed"), doc("c", "processing"), doc("d", "ready")];
  assert.deepEqual(pruneSelection(["d", "b", "a", "gone", "c"], library), ["d", "a"]);
});

test("a selection that is still valid is returned untouched so React does not re-render for nothing", () => {
  const selected = ["a", "d"];
  assert.equal(pruneSelection(selected, [doc("a", "ready"), doc("d", "ready")]), selected);
  assert.deepEqual(pruneSelection([], []), []);
});

// ---------- how a failed action is reported ----------

test("a connection failure from any action becomes the shared 'unreachable' state; everything else keeps its own words", () => {
  const offline = Object.assign(new Error(UNREACHABLE_MESSAGE), { status: 0 });
  assert.deepEqual(failureReport(offline, "Could not answer that question."), { unreachable: true });
  const refused = Object.assign(new Error("Only PDF files are supported."), { status: 415 });
  assert.deepEqual(failureReport(refused, "x"), { unreachable: false, message: "Only PDF files are supported." });
  assert.deepEqual(failureReport(new Error(""), "Could not remove document."), { unreachable: false, message: "Could not remove document." });
  assert.deepEqual(failureReport("weird", "Could not remove document."), { unreachable: false, message: "Could not remove document." });
});

// ---------- a jump is for the document it was made on ----------

test("a jump made on one document is not carried into a different document's viewer", () => {
  const jump = { page: 6, bbox: null, nonce: 1 };
  assert.equal(jumpForDocument(jump, "iot", "market"), null); // 'page 6' of a 3-page file would be a failed page load
  assert.equal(jumpForDocument(jump, "iot", null), null);
  assert.equal(jumpForDocument(jump, "iot", "iot"), jump); // re-clicking the open document keeps its highlight alive
  assert.equal(jumpForDocument(null, "iot", "market"), null);
});

// ---------- what an upload that didn't go through says ----------

test("a duplicate is a calm notice; every real failure is an error", () => {
  const duplicate = Object.assign(new Error('This PDF is already in your library as "a.pdf".'), { status: 409 });
  assert.deepEqual(uploadNotice(duplicate), { tone: "notice", text: 'This PDF is already in your library as "a.pdf".' });
  for (const [status, text] of [[415, "This file isn't a PDF. Only PDF files are supported."], [413, "PDF exceeds the 50 MB upload limit."], [400, "This file is empty."], [0, UNREACHABLE_MESSAGE]]) {
    assert.deepEqual(uploadNotice(Object.assign(new Error(text), { status })), { tone: "error", text });
  }
});

test("an upload failure of unknown shape still produces a sentence", () => {
  assert.deepEqual(uploadNotice("weird"), { tone: "error", text: "The upload failed. Please try again." });
  assert.deepEqual(uploadNotice(new Error("")), { tone: "error", text: "The upload failed. Please try again." });
  assert.equal(uploadNotice(new Error("disk full")).text, "disk full");
});

// ---------- polling ----------

test("suggested questions are re-requested only while pending, and only a bounded number of times", () => {
  assert.equal(nextSuggestionPoll(0, true), SUGGESTION_POLL_MS);
  assert.equal(nextSuggestionPoll(SUGGESTION_POLL_LIMIT - 1, true), SUGGESTION_POLL_MS);
  assert.equal(nextSuggestionPoll(SUGGESTION_POLL_LIMIT, true), null); // a generation that never finishes must not poll forever
  assert.equal(nextSuggestionPoll(0, false), null);
  assert.equal(nextSuggestionPoll(0, undefined), null);
});

test("the library is polled fast while something processes, slowly while the backend is down, and not at all otherwise", () => {
  assert.equal(refreshInterval([doc("a", "ready"), doc("b", "empty")], false), null);
  assert.equal(refreshInterval([doc("a", "ready"), doc("b", "processing")], false), 2000);
  assert.equal(refreshInterval([], false), null);
  assert.equal(refreshInterval([doc("a", "ready")], true), 5000);
  assert.equal(refreshInterval([doc("a", "processing")], true), 5000); // unreachable wins: a fast poll of a dead backend helps nobody
});

// ---------- the local-AI status card ----------

const ollama = (overrides = {}) => ({
  ollama: { available: true, generation_model_ready: true, embedding_model_ready: true, generation_model: "qwen3:8b", embedding_model: "qwen3-embedding:0.6b", ...overrides },
});

test("the status card says which part of the local AI is the problem", () => {
  assert.equal(systemView(ollama(), false).ready, true);
  assert.equal(systemView(null, false).headline, "Checking…");
  assert.match(systemView(null, true).headline, /Backend not reachable/);
  assert.match(systemView(ollama({ available: false }), false).headline, /Ollama isn't running/);
  assert.match(systemView(ollama({ generation_model_ready: false }), false).detail, /qwen3:8b/);
  assert.match(systemView(ollama({ embedding_model_ready: false }), false).detail, /qwen3-embedding:0.6b/);
  const both = systemView(ollama({ generation_model_ready: false, embedding_model_ready: false }), false).detail;
  assert.match(both, /qwen3:8b, qwen3-embedding:0.6b/);
  assert.equal(systemView(ollama({ generation_model_ready: false }), false).ready, false);
});

test("an older backend that doesn't name its models still gets a usable message", () => {
  const legacy = { ollama: { available: true, generation_model_ready: false, embedding_model_ready: true } };
  assert.match(systemView(legacy, false).detail, /answer model/);
});
