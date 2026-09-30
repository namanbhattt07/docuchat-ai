// Group 6 tutor session logic (lib/tutor.ts). Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { buildTutorRequest, EMPTY_TUTOR_SESSION, isActionAvailable, TUTOR_ACTIONS, tutorSessionAfter } from "../lib/tutor.ts";

const scope = { documentIds: ["doc-1", "doc-2"], collectionId: "col-1", groundingMode: "document" };
const action = id => TUTOR_ACTIONS.find(item => item.id === id);

test("all ten tutor actions are offered as controls", () => {
  assert.deepEqual(TUTOR_ACTIONS.map(item => item.id), [
    "explain_simply", "explain_deeply", "give_example", "give_analogy", "why", "compare", "quiz_me", "give_hint", "revise", "teach_from_beginning",
  ]);
});

test("an action button sends a structured tutor_action with the current topic", () => {
  const request = buildTutorRequest({ ...EMPTY_TUTOR_SESSION, topic: "ZigBee" }, scope, { action: "explain_simply" });
  assert.equal(request.question, "Explain simply");
  assert.equal(request.displayText, "Explain simply · ZigBee");
  assert.deepEqual(request.documentIds, ["doc-1", "doc-2"]);
  assert.equal(request.options.mode, "tutor");
  assert.equal(request.options.tutorAction, "explain_simply");
  assert.equal(request.options.currentTopic, "ZigBee");
  assert.equal(request.options.collectionId, "col-1");
});

test("typed text plus an action button becomes the new topic", () => {
  const request = buildTutorRequest({ ...EMPTY_TUTOR_SESSION, topic: "ZigBee" }, scope, { action: "give_analogy", typedText: "  Bluetooth  " });
  assert.equal(request.options.currentTopic, "Bluetooth");
  assert.equal(request.question, "Give analogy");
});

test("a typed reply while the tutor waits is sent as an answer to its question", () => {
  const state = { ...EMPTY_TUTOR_SESSION, topic: "ZigBee", pendingQuestion: "What topology does ZigBee use?" };
  const request = buildTutorRequest(state, scope, { typedText: "Mesh" });
  assert.equal(request.question, "Mesh");
  assert.equal(request.options.tutorAction, "check_answer");
  assert.equal(request.options.pendingQuestion, "What topology does ZigBee use?");
});

test("a pinned selection scopes the request to its own document and page", () => {
  const state = { ...EMPTY_TUTOR_SESSION, selection: { text: "QoS 2 is exactly-once.", page: 5, documentId: "doc-9" } };
  const request = buildTutorRequest(state, scope, { action: "explain_deeply" });
  assert.deepEqual(request.documentIds, ["doc-9"]);
  assert.equal(request.options.selectedText, "QoS 2 is exactly-once.");
  assert.equal(request.options.selectionPage, 5);
  assert.equal(request.options.collectionId, undefined);
  assert.equal(request.displayText, "Explain deeply · selected passage");
});

test("context-dependent actions are unavailable until there is something to act on", () => {
  assert.equal(isActionAvailable(action("why"), EMPTY_TUTOR_SESSION), false);
  assert.equal(isActionAvailable(action("why"), EMPTY_TUTOR_SESSION, "MQTT"), true);
  assert.equal(isActionAvailable(action("compare"), { ...EMPTY_TUTOR_SESSION, topic: "MQTT" }), true);
  assert.equal(isActionAvailable(action("teach_from_beginning"), EMPTY_TUTOR_SESSION), true);
});

test("the session follows the tutor's reply: topic and open question", () => {
  const result = { conversation_id: "c", answer: "", citations: [], tutor: { topic: "ZigBee", awaiting_answer: true, pending_question: "Name a ZigBee device type." } };
  const next = tutorSessionAfter(EMPTY_TUTOR_SESSION, result);
  assert.equal(next.topic, "ZigBee");
  assert.equal(next.pendingQuestion, "Name a ZigBee device type.");

  const answered = tutorSessionAfter(next, { ...result, tutor: { ...result.tutor, awaiting_answer: false, pending_question: null } });
  assert.equal(answered.pendingQuestion, null);
  assert.equal(answered.topic, "ZigBee");
});

test("non-tutor replies leave the tutor session untouched", () => {
  const state = { ...EMPTY_TUTOR_SESSION, topic: "ZigBee" };
  assert.equal(tutorSessionAfter(state, { conversation_id: "c", answer: "x", citations: [] }), state);
});
