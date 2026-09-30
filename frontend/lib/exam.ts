import type { AnswerEvaluation, ExamProgress, ExamQuestion, RequestedQuestionType, WritingFeedbackResult } from "./api";

// Group 6 EXAM / STUDY MODE: session-local study state. Everything here lives
// only in the browser tab (page.tsx holds it in a useReducer) -- the backend
// grades answers and computes the next ExamProgress, but never stores an exam.
// Pure reducer + selectors, no React/fetch, so it's unit-testable
// (see tests/exam.test.mjs).

export type StudyView = "exam" | "bank" | "writing";
export type ExamConfig = { questionType: RequestedQuestionType; count: number; topic: string; useSelection: boolean };
export type ExamAnswerRecord = { selected: string; evaluation: AnswerEvaluation };
export type ExamSession = {
  questions: ExamQuestion[];
  currentIndex: number;
  // The answer being composed for the current question, before submit --
  // kept here so switching modes mid-question doesn't lose it.
  draft: string;
  answers: Record<string, ExamAnswerRecord>;
  progress: ExamProgress;
  completed: boolean;
  scopeLabel: string;
  warnings: string[];
};
export type QuestionBank = { questions: ExamQuestion[]; warnings: string[]; scopeLabel: string };
export type WritingState = { question: string; answer: string; result: WritingFeedbackResult | null };
export type StudyState = { view: StudyView; config: ExamConfig; exam: ExamSession | null; bank: QuestionBank | null; writing: WritingState };

export const MAX_QUESTIONS = 20;
export const INITIAL_STUDY_STATE: StudyState = {
  view: "exam",
  config: { questionType: "mcq", count: 5, topic: "", useSelection: false },
  exam: null,
  bank: null,
  writing: { question: "", answer: "", result: null },
};

export type StudyAction =
  | { type: "view"; view: StudyView }
  | { type: "config"; patch: Partial<ExamConfig> }
  | { type: "exam/start"; questions: ExamQuestion[]; warnings: string[]; scopeLabel: string }
  | { type: "exam/draft"; value: string }
  | { type: "exam/answered"; questionId: string; selected: string; evaluation: AnswerEvaluation; progress: ExamProgress }
  | { type: "exam/next" }
  | { type: "exam/finish" }
  | { type: "exam/restart" }
  | { type: "exam/discard" }
  | { type: "bank/set"; questions: ExamQuestion[]; warnings: string[]; scopeLabel: string }
  | { type: "bank/practice" }
  | { type: "writing/update"; patch: Partial<Pick<WritingState, "question" | "answer">> }
  | { type: "writing/result"; result: WritingFeedbackResult }
  | { type: "writing/prefill"; question: string; answer: string };

export const clampCount = (value: number) => Math.min(MAX_QUESTIONS, Math.max(1, Math.round(Number.isFinite(value) ? value : 1)));

export function initialProgress(total: number): ExamProgress {
  return { total_questions: total, answered_questions: 0, correct_answers: 0, partial_answers: 0, score: 0, percentage: 0, completed: false, answered_ids: [] };
}

function newSession(questions: ExamQuestion[], warnings: string[], scopeLabel: string): ExamSession {
  return { questions, currentIndex: 0, draft: "", answers: {}, progress: initialProgress(questions.length), completed: false, scopeLabel, warnings };
}

export function studyReducer(state: StudyState, action: StudyAction): StudyState {
  const exam = state.exam;
  switch (action.type) {
    case "view":
      return { ...state, view: action.view };
    case "config":
      return { ...state, config: { ...state.config, ...action.patch, ...(action.patch.count !== undefined ? { count: clampCount(action.patch.count) } : {}) } };
    case "exam/start":
      if (action.questions.length === 0) return state;
      return { ...state, view: "exam", exam: newSession(action.questions, action.warnings, action.scopeLabel) };
    case "exam/draft":
      return exam ? { ...state, exam: { ...exam, draft: action.value } } : state;
    case "exam/answered":
      if (!exam || exam.answers[action.questionId]) return state; // a double-submit can't re-grade
      return {
        ...state,
        exam: { ...exam, answers: { ...exam.answers, [action.questionId]: { selected: action.selected, evaluation: action.evaluation } }, progress: action.progress },
      };
    case "exam/next": {
      if (!exam) return state;
      const next = exam.currentIndex + 1;
      return next >= exam.questions.length
        ? { ...state, exam: { ...exam, completed: true, draft: "" } }
        : { ...state, exam: { ...exam, currentIndex: next, draft: "" } };
    }
    case "exam/finish":
      return exam ? { ...state, exam: { ...exam, completed: true, draft: "" } } : state;
    case "exam/restart":
      return exam ? { ...state, exam: newSession(exam.questions, exam.warnings, exam.scopeLabel) } : state;
    case "exam/discard":
      return { ...state, exam: null };
    case "bank/set":
      return { ...state, view: "bank", bank: { questions: action.questions, warnings: action.warnings, scopeLabel: action.scopeLabel } };
    case "bank/practice":
      return state.bank && state.bank.questions.length
        ? { ...state, view: "exam", exam: newSession(state.bank.questions, state.bank.warnings, state.bank.scopeLabel) }
        : state;
    case "writing/update":
      return { ...state, writing: { ...state.writing, ...action.patch } };
    case "writing/result":
      // The original answer stays in `writing.answer` untouched; the
      // suggestion only ever lives inside `result`.
      return { ...state, writing: { ...state.writing, result: action.result } };
    case "writing/prefill":
      return { ...state, view: "writing", writing: { question: action.question, answer: action.answer, result: null } };
    default:
      return state;
  }
}

export const questionText = (question: ExamQuestion) => (question.type === "true_false" ? question.statement : question.question);

export function correctAnswerText(question: ExamQuestion): string {
  if (question.type === "true_false") return question.answer ? "True" : "False";
  if (question.type === "mcq") return question.answer;
  return question.expected_answer;
}

export const QUESTION_TYPE_LABELS: Record<ExamQuestion["type"], string> = { mcq: "Multiple choice", true_false: "True / False", short_answer: "Short answer" };

export function currentQuestion(session: ExamSession): ExamQuestion | null {
  return session.completed ? null : session.questions[session.currentIndex] ?? null;
}

export type ExamSummary = { total: number; answered: number; correct: number; partial: number; incorrect: number; unanswered: number; score: number; percentage: number };

export function examSummary(session: ExamSession): ExamSummary {
  const { progress } = session;
  const incorrect = progress.answered_questions - progress.correct_answers - progress.partial_answers;
  return {
    total: progress.total_questions,
    answered: progress.answered_questions,
    correct: progress.correct_answers,
    partial: progress.partial_answers,
    incorrect,
    unanswered: progress.total_questions - progress.answered_questions,
    score: progress.score,
    percentage: progress.percentage,
  };
}
