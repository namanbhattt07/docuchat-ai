const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

// Every failed request surfaces as an ApiError whose `message` is fit to show a
// person as-is. `status` 0 means the backend could not be reached at all (not
// running, wrong address, blocked by CORS) -- the browser only says "Failed to
// fetch" for those, which tells nobody what to do.
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export const UNREACHABLE_MESSAGE = "Can't reach the DocuChat backend. Make sure it is running, then try again.";
export const isConnectionError = (error: unknown): boolean => error instanceof ApiError && error.status === 0;

function failureMessage(status: number, detail: unknown): string {
  // FastAPI validation errors arrive as a list of {msg} objects rather than a
  // string -- surface their messages instead of "[object Object]".
  if (Array.isArray(detail)) {
    const joined = detail.map((item: { msg?: string }) => item?.msg).filter(Boolean).join("; ");
    if (joined) return joined;
  }
  if (typeof detail === "string" && detail.trim()) return detail;
  return status >= 500 ? "The server hit an unexpected error. Please try again in a moment." : "Something went wrong. Please try again.";
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new ApiError(UNREACHABLE_MESSAGE, 0);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(failureMessage(response.status, body?.detail), response.status);
  }
  return response.status === 204 ? (undefined as T) : response.json() as Promise<T>;
}

export type DocumentStatus = "processing" | "ready" | "empty" | "failed";
// Group 7 explicit per-page processing outcome (backend document_model.PAGE_*):
// "unknown" only for pages indexed before per-page status existed.
export type PageStatus = "text" | "ocr" | "empty" | "failed" | "unknown";
export type PageSummary = Record<PageStatus, number>;
export type DocumentItem = {
  id: string; filename: string; page_count: number; status: DocumentStatus; status_detail: string | null; created_at: string;
  page_summary?: PageSummary;
};
export type BBox = [number, number, number, number];
export type BboxSource = "exact" | "approximate" | "unavailable";
// "ocr": recognized (scanned) text; "figure": a detected figure/image on the page.
export type CitationSourceType = "text" | "ocr" | "figure" | "selection";
export type Citation = {
  index: number; document_id: string; filename: string; page_number: number; excerpt: string; section?: string | null;
  bbox?: BBox | null; bbox_source?: BboxSource; source_type?: CitationSourceType;
};
export type GroundingMode = "document" | "free";

// Group 6 learning modes (+ Group 7 "template"). "chat"/"tutor"/"brainstorm"/"template" all go through /chat
// (see backend ChatRequest.mode); "exam" is UI-only -- its requests go to
// /learning/* and nothing about an exam session is persisted server-side.
export type ChatMode = "chat" | "tutor" | "brainstorm" | "template";
export type AssistantMode = ChatMode | "exam";
export type TutorActionId =
  | "explain_simply" | "explain_deeply" | "give_example" | "give_analogy" | "why" | "compare"
  | "quiz_me" | "give_hint" | "revise" | "teach_from_beginning" | "check_answer";
export type TutorSectionKind = "document" | "tutor" | "example" | "analogy" | "question" | "notice";
export type TutorSection = { kind: TutorSectionKind; title: string; content: string; note: string };
export type TutorMeta = {
  topic: string; action: TutorActionId | null; action_label: string | null; sections: TutorSection[];
  awaiting_answer: boolean; pending_question: string | null; grounded: boolean; insufficient_evidence: boolean; grounding_mode: GroundingMode;
};
export type BrainstormEvidence = { point: string; source_ids: number[] };
export type BrainstormIdea = { title: string; description: string; kind: string; builds_on: number[] };
export type BrainstormMeta = { evidence: BrainstormEvidence[]; ideas: BrainstormIdea[]; notice: string; insufficient_evidence: boolean; grounding_mode: GroundingMode };
// Group 7 visual Q&A. `supported` is only true when a real local vision model
// answered; otherwise the answer is the honest "not supported" reply and the
// targets are the pages/figures the user can preview and open.
export type VisualFigure = { id: string; kind: "image" | "figure_caption"; label: string | null; caption: string | null; bbox: BBox | null; preview_url: string };
export type VisualTarget = { document_id: string; filename: string; page_number: number; page_image_url: string; figures: VisualFigure[] };
export type VisualMeta = { supported: boolean; model: string | null; reason: string; targets: VisualTarget[] };
export type ChatResult = {
  conversation_id: string; answer: string; citations: Citation[]; message_id?: string;
  mode?: ChatMode; tutor?: TutorMeta; brainstorm?: BrainstormMeta; visual?: VisualMeta; template?: { id: string; name: string };
};

export type QuestionType = "mcq" | "true_false" | "short_answer";
export type RequestedQuestionType = QuestionType | "mixed";
type QuestionBase = { id: string; explanation: string; source_ids: number[]; citations: Citation[]; grounded: boolean };
export type MCQQuestion = QuestionBase & { type: "mcq"; question: string; options: string[]; answer: string };
export type TrueFalseQuestion = QuestionBase & { type: "true_false"; statement: string; answer: boolean };
export type ShortAnswerQuestion = QuestionBase & { type: "short_answer"; question: string; expected_answer: string; key_points: string[] };
export type ExamQuestion = MCQQuestion | TrueFalseQuestion | ShortAnswerQuestion;
export type QuestionSetResult = { questions: ExamQuestion[]; question_type: RequestedQuestionType; requested: number; generated: number; grounding_mode: GroundingMode; warnings: string[] };
export type AnswerVerdict = "correct" | "partial" | "incorrect";
export type AnswerEvaluation = {
  question_id: string; question_type: QuestionType; verdict: AnswerVerdict; points: number; max_points: number;
  selected_answer: string; correct_answer: string; feedback: string; explanation: string;
  key_points_covered: string[]; key_points_missing: string[]; method: "exact" | "model" | "keyword_fallback"; citations: Citation[];
};
export type ExamProgress = {
  total_questions: number; answered_questions: number; correct_answers: number; partial_answers: number;
  score: number; percentage: number; completed: boolean; answered_ids: string[];
};
export type RubricRating = "strong" | "adequate" | "needs_work" | "not_assessed";
export type WritingFeedbackResult = {
  original_answer: string;
  suggested_answer: string;
  feedback: {
    suggested_answer: string; summary: string; strengths: string[]; improvements: string[];
    missing_points: { point: string; source_ids: number[] }[]; factual_issues: { issue: string; source_ids: number[] }[];
    rubric: { criterion: string; label: string; rating: RubricRating; comment: string }[]; structured: boolean;
  };
  citations: Citation[];
  grounding_mode: GroundingMode;
  grounded: boolean;
};
// The document/collection scope + grounding every learning request shares.
export type LearningScope = { documentIds: string[]; collectionId?: string; groundingMode: GroundingMode; selectedText?: string; selectionPage?: number };

// Group 5: multi-document collections. A collection's `documents` are
// summaries (see backend services/collections.py::_document_summary) --
// enough for sidebar rendering without a second round trip per document.
export type CollectionDocumentSummary = { id: string; filename: string; page_count: number; status: DocumentStatus };
export type CollectionItem = { id: string; name: string; created_at: string; updated_at: string; documents: CollectionDocumentSummary[] };

export type TocSource = "native" | "heuristic";
export type TocItem = { id: string; title: string; page: number; level: number; source: TocSource };
export type TocResponse = { document_id: string; title: string; items: TocItem[] };

export type SearchMatch = { index: number; page: number; start_offset: number; end_offset: number; bbox: BBox | null; snippet: string };
export type SearchResponse = { query: string; total: number; matches: SearchMatch[] };

// `pending`: the document is ready but its suggested questions are still being written -- ask again shortly.
export type SuggestedQuestionsResponse = { document_id: string; questions: string[]; pending?: boolean };

export type SelectionAction = "explain" | "summarize" | "analyze" | "rewrite";

export { API_BASE_URL };
export const pdfFileUrl = (documentId: string) => `${API_BASE_URL}/api/v1/documents/${documentId}/file`;
export const pdfDownloadUrl = (documentId: string) => `${pdfFileUrl(documentId)}?download=1`;

export const listDocuments = () => request<DocumentItem[]>("/api/v1/documents");
export type SystemStatus = {
  ollama: { available: boolean; generation_model_ready: boolean; embedding_model_ready: boolean; generation_model?: string; embedding_model?: string };
  ocr?: { enabled: boolean; available: boolean; engine: string; detail: string | null };
  vision?: { supported: boolean; model: string | null; reason: string };
};
export const getSystemStatus = () => request<SystemStatus>("/api/v1/system/status");
export async function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentItem>("/api/v1/documents", { method: "POST", body: form });
}
export const deleteDocument = (id: string) => request<void>(`/api/v1/documents/${id}`, { method: "DELETE" });
// Group 5: collection scope, grounding mode, and reply-to-message-id are all
// optional extras layered on the existing /chat contract (see backend
// ChatRequest) -- an options bag keeps every pre-Group-5 call site working
// unchanged rather than growing a long positional parameter list.
export type AskQuestionOptions = {
  collectionId?: string; groundingMode?: GroundingMode; replyToMessageId?: string;
  // Group 6: tutor/brainstorm ride on the same /chat request.
  mode?: ChatMode; tutorAction?: TutorActionId; currentTopic?: string; pendingQuestion?: string;
  selectedText?: string; selectionPage?: number;
  // Group 7: which prompt template to apply (mode "template").
  templateId?: string;
};
export const askQuestion = (question: string, document_ids: string[], conversation_id?: string, options?: AskQuestionOptions) =>
  request<ChatResult>("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      document_ids,
      conversation_id,
      collection_id: options?.collectionId,
      grounding_mode: options?.groundingMode,
      reply_to_message_id: options?.replyToMessageId,
      mode: options?.mode,
      tutor_action: options?.tutorAction,
      current_topic: options?.currentTopic,
      pending_question: options?.pendingQuestion,
      selected_text: options?.selectedText,
      selection_page: options?.selectionPage,
      template_id: options?.templateId,
    }),
  });
// Selection-based AI (Group 4): the selected passage is sent as explicit,
// primary context (selected_text/selection_page/selection_action), not left
// for retrieval to rediscover -- see backend ChatRequest/RetrievalMode.SELECTION.
const SELECTION_ACTION_QUESTIONS: Record<SelectionAction, string> = {
  explain: "Explain the selected passage.",
  summarize: "Summarize the selected passage.",
  analyze: "Analyze the selected passage.",
  rewrite: "Rewrite the selected passage.",
};
export const askSelectionAction = (action: SelectionAction, selectedText: string, documentIds: string[], selectionPage?: number, conversationId?: string) =>
  request<ChatResult>("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: SELECTION_ACTION_QUESTIONS[action],
      document_ids: documentIds,
      conversation_id: conversationId,
      selected_text: selectedText,
      selection_page: selectionPage,
      selection_action: action,
    }),
  });
export const generateAbstract = (document_ids: string[], conversation_id?: string) => request<ChatResult>("/api/v1/chat/abstract", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document_ids, conversation_id }) });
export const getDocumentToc = (id: string) => request<TocResponse>(`/api/v1/documents/${id}/toc`);
export const searchDocument = (id: string, query: string) => request<SearchResponse>(`/api/v1/documents/${id}/search?q=${encodeURIComponent(query)}`);
export const getSuggestedQuestions = (id: string) => request<SuggestedQuestionsResponse>(`/api/v1/documents/${id}/suggested-questions`);

// Group 5: multi-document collections CRUD (see backend api/v1/collections.py).
export const listCollections = () => request<CollectionItem[]>("/api/v1/collections");
export const createCollection = (name: string) => request<CollectionItem>("/api/v1/collections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
export const renameCollection = (id: string, name: string) => request<CollectionItem>(`/api/v1/collections/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
export const deleteCollection = (id: string) => request<void>(`/api/v1/collections/${id}`, { method: "DELETE" });
export const addDocumentToCollection = (collectionId: string, documentId: string) => request<CollectionItem>(`/api/v1/collections/${collectionId}/documents`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document_id: documentId }) });
export const removeDocumentFromCollection = (collectionId: string, documentId: string) => request<CollectionItem>(`/api/v1/collections/${collectionId}/documents/${documentId}`, { method: "DELETE" });

// Group 6: exam / question bank / writing feedback (see backend api/v1/learning.py).
const learningScopeBody = (scope: LearningScope) => ({
  document_ids: scope.documentIds,
  collection_id: scope.collectionId,
  grounding_mode: scope.groundingMode,
  selected_text: scope.selectedText,
  selection_page: scope.selectionPage,
});
export type GenerateQuestionsParams = { questionType: RequestedQuestionType; count: number; topic?: string; seed?: number };
export const generateQuestions = (scope: LearningScope, params: GenerateQuestionsParams) =>
  request<QuestionSetResult>("/api/v1/learning/questions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...learningScopeBody(scope), question_type: params.questionType, count: params.count, topic: params.topic || undefined, seed: params.seed }),
  });
export const evaluateExamAnswer = (question: ExamQuestion, answer: string, progress?: ExamProgress) =>
  request<{ evaluation: AnswerEvaluation; progress: ExamProgress | null }>("/api/v1/learning/exam/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, answer, progress }),
  });
export const evaluateWriting = (scope: LearningScope, answer: string, question?: string) =>
  request<WritingFeedbackResult>("/api/v1/learning/writing/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...learningScopeBody(scope), answer, question: question || undefined }),
  });

// Group 7: OCR page status, page/figure images (see backend api/v1/documents.py).
export type PageInfo = {
  page_number: number; status: PageStatus; source_type: "text" | "ocr" | null; status_detail: string | null;
  ocr_confidence: number | null; char_count: number; figure_count: number;
};
export type DocumentPagesResponse = { document_id: string; page_count: number; pages: PageInfo[] };
export type DocumentImageItem = {
  id: string; document_id: string; page_number: number; kind: "image" | "figure_caption"; label: string | null; caption: string | null;
  bbox: BBox | null; width: number | null; height: number | null; xref: number | null; preview_url: string; page_image_url: string;
};
export type DocumentImagesResponse = { document_id: string; total: number; images: DocumentImageItem[] };
export const getDocumentPages = (id: string) => request<DocumentPagesResponse>(`/api/v1/documents/${id}/pages`);
export const getDocumentImages = (id: string) => request<DocumentImagesResponse>(`/api/v1/documents/${id}/images`);
// Backend returns root-relative URLs; images are fetched straight from the API origin.
export const assetUrl = (path: string) => `${API_BASE_URL}${path}`;

// Group 7: prompt templates (see backend api/v1/templates.py). Applying one
// is just askQuestion(..., { mode: "template", templateId }).
export type TemplateItem = {
  id: string; name: string; description: string; instruction: string; output_format: string | null; builtin: boolean;
  created_at: string | null; updated_at: string | null;
};
export type TemplateDraft = { name: string; description: string; instruction: string; output_format: string };
const templateBody = (draft: TemplateDraft) => JSON.stringify({
  name: draft.name, description: draft.description || null, instruction: draft.instruction, output_format: draft.output_format || null,
});
export const listTemplates = () => request<{ templates: TemplateItem[] }>("/api/v1/templates").then(result => result.templates);
export const createTemplate = (draft: TemplateDraft) => request<TemplateItem>("/api/v1/templates", { method: "POST", headers: { "Content-Type": "application/json" }, body: templateBody(draft) });
export const updateTemplate = (id: string, draft: TemplateDraft) => request<TemplateItem>(`/api/v1/templates/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: templateBody(draft) });
export const deleteTemplate = (id: string) => request<void>(`/api/v1/templates/${id}`, { method: "DELETE" });
