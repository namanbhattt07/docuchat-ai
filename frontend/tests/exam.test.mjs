// Group 6 exam/study session state (lib/exam.ts). Run with `npm test`
// (Node's built-in runner; Node >= 22.6 strips the TypeScript types).
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  clampCount, correctAnswerText, currentQuestion, examSummary, INITIAL_STUDY_STATE, initialProgress, questionText, studyReducer,
} from "../lib/exam.ts";

const mcq = { id: "q1", type: "mcq", question: "Which layer?", options: ["App", "Link", "Net", "Phys"], answer: "App", explanation: "", source_ids: [1], citations: [], grounded: true };
const tf = { id: "q2", type: "true_false", statement: "MQTT uses a broker.", answer: true, explanation: "", source_ids: [1], citations: [], grounded: true };
const short = { id: "q3", type: "short_answer", question: "What does a broker do?", expected_answer: "Routes messages.", key_points: ["routes messages"], explanation: "", source_ids: [1], citations: [], grounded: true };

const evaluation = (questionId, verdict, points) => ({
  question_id: questionId, question_type: "mcq", verdict, points, max_points: 1, selected_answer: "App", correct_answer: "App",
  feedback: "", explanation: "", key_points_covered: [], key_points_missing: [], method: "exact", citations: [],
});
const progress = (answered, correct, score, completed = false) => ({
  ...initialProgress(3), answered_questions: answered, correct_answers: correct, score, percentage: answered ? (100 * score) / answered : 0, completed,
});

function startedState() {
  return studyReducer(INITIAL_STUDY_STATE, { type: "exam/start", questions: [mcq, tf, short], warnings: [], scopeLabel: "mqtt.pdf" });
}

test("starting an exam creates a fresh session-local progress", () => {
  const state = startedState();
  assert.equal(state.view, "exam");
  assert.equal(state.exam.currentIndex, 0);
  assert.deepEqual(state.exam.progress, initialProgress(3));
  assert.equal(currentQuestion(state.exam).id, "q1");
});

test("an empty question set never starts an exam", () => {
  const state = studyReducer(INITIAL_STUDY_STATE, { type: "exam/start", questions: [], warnings: [], scopeLabel: "x" });
  assert.equal(state.exam, null);
});

test("answer -> next -> results flow tracks progress from the server", () => {
  let state = startedState();
  state = studyReducer(state, { type: "exam/draft", value: "App" });
  assert.equal(state.exam.draft, "App");
  state = studyReducer(state, { type: "exam/answered", questionId: "q1", selected: "App", evaluation: evaluation("q1", "correct", 1), progress: progress(1, 1, 1) });
  assert.equal(state.exam.answers.q1.selected, "App");
  assert.equal(state.exam.progress.answered_questions, 1);

  state = studyReducer(state, { type: "exam/next" });
  assert.equal(state.exam.currentIndex, 1);
  assert.equal(state.exam.draft, "", "the draft resets for the next question");

  state = studyReducer(state, { type: "exam/next" });
  state = studyReducer(state, { type: "exam/next" });
  assert.equal(state.exam.completed, true);
  assert.equal(currentQuestion(state.exam), null);
});

test("a double submit cannot re-grade an answered question", () => {
  let state = startedState();
  state = studyReducer(state, { type: "exam/answered", questionId: "q1", selected: "App", evaluation: evaluation("q1", "correct", 1), progress: progress(1, 1, 1) });
  const again = studyReducer(state, { type: "exam/answered", questionId: "q1", selected: "Link", evaluation: evaluation("q1", "incorrect", 0), progress: progress(2, 1, 1) });
  assert.equal(again, state);
});

test("finish early, then restart with the same questions and a clean slate", () => {
  let state = startedState();
  state = studyReducer(state, { type: "exam/answered", questionId: "q1", selected: "App", evaluation: evaluation("q1", "correct", 1), progress: progress(1, 1, 1) });
  state = studyReducer(state, { type: "exam/finish" });
  const summary = examSummary(state.exam);
  assert.deepEqual([summary.answered, summary.correct, summary.unanswered, summary.score], [1, 1, 2, 1]);

  state = studyReducer(state, { type: "exam/restart" });
  assert.equal(state.exam.completed, false);
  assert.deepEqual(state.exam.answers, {});
  assert.deepEqual(state.exam.questions.map(q => q.id), ["q1", "q2", "q3"]);
  assert.deepEqual(state.exam.progress, initialProgress(3));

  state = studyReducer(state, { type: "exam/discard" });
  assert.equal(state.exam, null);
});

test("summary counts incorrect answers from the progress numbers", () => {
  let state = startedState();
  state = studyReducer(state, {
    type: "exam/answered", questionId: "q1", selected: "Link", evaluation: evaluation("q1", "incorrect", 0),
    progress: { ...initialProgress(3), answered_questions: 2, correct_answers: 0, partial_answers: 1, score: 0.5, percentage: 25 },
  });
  const summary = examSummary(state.exam);
  assert.equal(summary.incorrect, 1);
  assert.equal(summary.partial, 1);
});

test("question bank can be practised as an exam", () => {
  let state = studyReducer(INITIAL_STUDY_STATE, { type: "bank/set", questions: [tf, short], warnings: ["w"], scopeLabel: "doc" });
  assert.equal(state.view, "bank");
  state = studyReducer(state, { type: "bank/practice" });
  assert.equal(state.view, "exam");
  assert.equal(state.exam.questions.length, 2);
  assert.equal(state.exam.progress.total_questions, 2);
});

test("writing feedback keeps the original answer separate from the suggestion", () => {
  let state = studyReducer(INITIAL_STUDY_STATE, { type: "writing/prefill", question: "What is MQTT?", answer: "My original words." });
  assert.equal(state.view, "writing");
  const result = { original_answer: "My original words.", suggested_answer: "A better version.", feedback: {}, citations: [], grounding_mode: "document", grounded: true };
  state = studyReducer(state, { type: "writing/result", result });
  assert.equal(state.writing.answer, "My original words.");
  assert.equal(state.writing.result.suggested_answer, "A better version.");
});

test("question count is clamped to the supported range", () => {
  assert.equal(clampCount(0), 1);
  assert.equal(clampCount(55), 20);
  assert.equal(clampCount(Number.NaN), 1);
  const state = studyReducer(INITIAL_STUDY_STATE, { type: "config", patch: { count: 99 } });
  assert.equal(state.config.count, 20);
});

test("question text and correct answers per type", () => {
  assert.equal(questionText(tf), "MQTT uses a broker.");
  assert.equal(questionText(mcq), "Which layer?");
  assert.equal(correctAnswerText(tf), "True");
  assert.equal(correctAnswerText(mcq), "App");
  assert.equal(correctAnswerText(short), "Routes messages.");
});
