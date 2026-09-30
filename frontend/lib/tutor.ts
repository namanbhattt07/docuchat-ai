import type { AskQuestionOptions, ChatResult, GroundingMode, TutorActionId } from "./api";

// Group 6 TUTOR MODE: the client-held half of the tutor session (see
// backend services/tutor.py::TutorSession). Pure functions only -- no React,
// no fetch -- so page.tsx just stores the state and ChatPanel just renders it.

export type TutorSelection = { text: string; page: number; documentId: string };
export type TutorSessionState = {
  topic: string | null;
  // The question the tutor last asked (Quiz me, or a "check your
  // understanding" prompt). While set, the student's next typed message is
  // sent as an answer to it.
  pendingQuestion: string | null;
  selection: TutorSelection | null;
};
export const EMPTY_TUTOR_SESSION: TutorSessionState = { topic: null, pendingQuestion: null, selection: null };

type TutorActionDefinition = { id: Exclude<TutorActionId, "check_answer">; label: string; title: string; needsContext?: boolean };

// Labels mirror backend TUTOR_ACTIONS so persisted history reads the same.
export const TUTOR_ACTIONS: TutorActionDefinition[] = [
  { id: "explain_simply", label: "Explain simply", title: "Plain-language explanation with a small example" },
  { id: "explain_deeply", label: "Explain deeply", title: "Detailed technical explanation: mechanism, assumptions, details" },
  { id: "give_example", label: "Give example", title: "A concrete example, from the document where possible" },
  { id: "give_analogy", label: "Give analogy", title: "An intuitive analogy, connected back to the real concept" },
  { id: "why", label: "Why?", title: "Why this exists and what problem it solves", needsContext: true },
  { id: "compare", label: "Compare", title: "Compare with the most relevant related concept", needsContext: true },
  { id: "quiz_me", label: "Quiz me", title: "One question — the tutor waits for your answer" },
  { id: "give_hint", label: "Give hint", title: "A hint that doesn't give the answer away", needsContext: true },
  { id: "revise", label: "Revise", title: "Concise revision summary of the topic" },
  { id: "teach_from_beginning", label: "Teach from beginning", title: "Start from prerequisites, step by step" },
];

export const tutorActionLabel = (id: TutorActionId) => TUTOR_ACTIONS.find(action => action.id === id)?.label ?? "Check my answer";

export function hasTutorContext(state: TutorSessionState, typedText = ""): boolean {
  return Boolean(state.topic || state.selection || state.pendingQuestion || typedText.trim());
}

export function isActionAvailable(action: TutorActionDefinition, state: TutorSessionState, typedText = ""): boolean {
  return !action.needsContext || hasTutorContext(state, typedText);
}

export type TutorRequest = { question: string; displayText: string; documentIds: string[]; options: AskQuestionOptions };

/**
 * Build the /chat request for a tutor turn.
 * - An action button with text in the composer treats that text as a new topic.
 * - A typed message while the tutor is waiting for an answer is sent as that answer.
 * - A selection pinned to the tutor always scopes the request to its own document.
 */
export function buildTutorRequest(
  state: TutorSessionState,
  scope: { documentIds: string[]; collectionId?: string; groundingMode: GroundingMode; replyToMessageId?: string },
  input: { action?: TutorActionId; typedText?: string },
): TutorRequest {
  const typed = (input.typedText ?? "").trim();
  const selection = state.selection;
  let action = input.action;
  let topic = state.topic ?? undefined;
  let question = typed;
  if (action) {
    if (typed) topic = typed;
    question = tutorActionLabel(action);
  } else if (state.pendingQuestion) {
    action = "check_answer";
  }
  const target = selection ? "selected passage" : topic;
  const displayText = input.action ? `${tutorActionLabel(input.action)}${target ? ` · ${target}` : ""}` : typed;
  return {
    question,
    displayText,
    documentIds: selection ? [selection.documentId] : scope.documentIds,
    options: {
      mode: "tutor",
      tutorAction: action,
      currentTopic: topic,
      pendingQuestion: state.pendingQuestion ?? undefined,
      selectedText: selection?.text,
      selectionPage: selection?.page,
      // A selection is validated against its own document, like Group 4's
      // selection actions -- not against the active collection.
      collectionId: selection ? undefined : scope.collectionId,
      groundingMode: scope.groundingMode,
      replyToMessageId: scope.replyToMessageId,
    },
  };
}

export function tutorSessionAfter(state: TutorSessionState, result: ChatResult): TutorSessionState {
  if (!result.tutor) return state;
  return {
    ...state,
    topic: result.tutor.topic || state.topic,
    pendingQuestion: result.tutor.awaiting_answer ? result.tutor.pending_question : null,
  };
}
