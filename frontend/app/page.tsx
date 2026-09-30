"use client";

import dynamic from "next/dynamic";
import { FormEvent, PointerEvent as ReactPointerEvent, useCallback, useEffect, useReducer, useState } from "react";
import {
  addDocumentToCollection, askQuestion, askSelectionAction, AssistantMode, BBox, ChatResult, Citation, CollectionItem, createCollection, deleteCollection,
  deleteDocument, DocumentItem, generateAbstract, getDocumentPages, getSuggestedQuestions, getSystemStatus, GroundingMode, LearningScope, listCollections,
  listDocuments, PageInfo, pdfDownloadUrl, pdfFileUrl, removeDocumentFromCollection, renameCollection, SystemStatus, TemplateItem, TutorActionId, UNREACHABLE_MESSAGE,
  uploadDocument,
} from "../lib/api";
import { failureReport, isChattable, jumpForDocument, nextSuggestionPoll, pruneSelection, refreshInterval, systemView, UploadNotice, uploadNotice } from "../lib/documents";
import { INITIAL_STUDY_STATE, studyReducer } from "../lib/exam";
import { pageChip } from "../lib/ocr";
import { buildTemplateRequest } from "../lib/templates";
import { buildTutorRequest, EMPTY_TUTOR_SESSION, TutorSessionState, tutorSessionAfter } from "../lib/tutor";
import ChatPanel, { Message, QuoteAction, ReplyTarget } from "./components/ChatPanel";
import ExamPanel from "./components/ExamPanel";
import FiguresPanel from "./components/FiguresPanel";
import { CloseIcon, FileIcon, ImageIcon, ListIcon } from "./components/icons";
import type { JumpTarget, SelectionPayload, TutorSelectionPayload } from "./components/PdfViewer";
import Sidebar from "./components/Sidebar";
import TemplatePanel from "./components/TemplatePanel";
import TocPanel from "./components/TocPanel";
import TutorControls from "./components/TutorControls";

// pdfjs-dist assumes a browser environment at import time (it reaches for
// `document`/canvas APIs while loading), which crashes Next.js's server-side
// prerender of this page. Loading the viewer client-only sidesteps that.
const PdfViewer = dynamic(() => import("./components/PdfViewer"), {
  ssr: false,
  loading: () => <div className="center-empty"><div className="center-empty-skeleton" /></div>,
});

type Theme = "light" | "dark";
type MobileView = "library" | "document" | "chat";

const SIDEBAR_MIN = 220;
const SIDEBAR_MAX = 420;
const SIDEBAR_DEFAULT = 292;
const CHAT_MIN = 300;
const CHAT_MAX = 520;
const CHAT_DEFAULT = 380;

function readStoredNumber(key: string, fallback: number): number {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    const parsed = raw ? Number(raw) : NaN;
    return Number.isFinite(parsed) ? parsed : fallback;
  } catch { return fallback; }
}
function readStoredBool(key: string, fallback: boolean): boolean {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? fallback : raw === "1";
  } catch { return fallback; }
}
function writeStored(key: string, value: string) {
  try { window.localStorage.setItem(key, value); } catch { /* private browsing or storage disabled -- layout just won't persist */ }
}
const assistantMessage = (result: ChatResult): Message => ({
  role: "assistant", content: result.answer, citations: result.citations, id: result.message_id, tutor: result.tutor, brainstorm: result.brainstorm,
  visual: result.visual, template: result.template,
});

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [conversationId, setConversationId] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [abstractLoading, setAbstractLoading] = useState(false);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState("");
  // Group 8: the backend being unreachable is a *state* (it clears itself when
  // the backend comes back), not a one-off error message that lingers.
  const [apiUnreachable, setApiUnreachable] = useState(false);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [notice, setNotice] = useState<UploadNotice | null>(null);
  const [theme, setTheme] = useState<Theme>("light");

  // Group 5: multi-document collections, grounding control, and reply
  // threading. `selected` doubles as "the effective document scope" for both
  // a plain ad-hoc multi-select (activeCollectionId null) and a collection's
  // checked members (activeCollectionId set) -- see selectCollection/toggleSelect.
  const [collections, setCollections] = useState<CollectionItem[]>([]);
  const [activeCollectionId, setActiveCollectionId] = useState<string | null>(null);
  const [groundingMode, setGroundingMode] = useState<GroundingMode>("document");
  const [replyTarget, setReplyTarget] = useState<ReplyTarget | null>(null);

  // Group 6 learning modes. The tutor session (topic / open question /
  // pinned selection) and the whole study state (exam, question bank,
  // writing feedback) are session-local: React state only, never persisted.
  const [mode, setMode] = useState<AssistantMode>("chat");
  const [tutorSession, setTutorSession] = useState<TutorSessionState>(EMPTY_TUTOR_SESSION);
  const [study, dispatchStudy] = useReducer(studyReducer, INITIAL_STUDY_STATE);

  const [openDocs, setOpenDocs] = useState<string[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [jumpTarget, setJumpTarget] = useState<JumpTarget | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [tocOpen, setTocOpen] = useState(false);
  // Group 7: figures/pages browser (shares the TOC's slide-over slot, so at
  // most one of the two is open) and the open document's per-page OCR status.
  const [figuresOpen, setFiguresOpen] = useState(false);
  const [pageInfos, setPageInfos] = useState<PageInfo[]>([]);

  // Layout preferences start at the defaults, which is all the server can know, and
  // are restored from localStorage after mount (like the theme below). Reading them
  // in the initial state made the first client render differ from the server's HTML
  // whenever something non-default was saved -- a hydration mismatch.
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [chatCollapsed, setChatCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT);
  const [chatWidth, setChatWidth] = useState(CHAT_DEFAULT);
  const [isPreviewVisible, setIsPreviewVisible] = useState(true);
  const [layoutRestored, setLayoutRestored] = useState(false);
  const [mobileView, setMobileView] = useState<MobileView>("library");

  useEffect(() => {
    const current = document.documentElement.getAttribute("data-theme");
    if (current === "dark" || current === "light") setTheme(current);
  }, []);
  useEffect(() => {
    setSidebarCollapsed(readStoredBool("docuchat-sidebar-collapsed", false));
    setChatCollapsed(readStoredBool("docuchat-chat-collapsed", false));
    setSidebarWidth(readStoredNumber("docuchat-sidebar-width", SIDEBAR_DEFAULT));
    setChatWidth(readStoredNumber("docuchat-chat-width", CHAT_DEFAULT));
    setIsPreviewVisible(readStoredBool("docuchat-preview-visible", true));
    setLayoutRestored(true);
  }, []);
  // Saving waits for the restore above, or the defaults would overwrite what was saved.
  useEffect(() => { if (layoutRestored) writeStored("docuchat-sidebar-collapsed", sidebarCollapsed ? "1" : "0"); }, [layoutRestored, sidebarCollapsed]);
  useEffect(() => { if (layoutRestored) writeStored("docuchat-chat-collapsed", chatCollapsed ? "1" : "0"); }, [layoutRestored, chatCollapsed]);
  useEffect(() => { if (layoutRestored) writeStored("docuchat-sidebar-width", String(sidebarWidth)); }, [layoutRestored, sidebarWidth]);
  useEffect(() => { if (layoutRestored) writeStored("docuchat-chat-width", String(chatWidth)); }, [layoutRestored, chatWidth]);
  useEffect(() => { if (layoutRestored) writeStored("docuchat-preview-visible", isPreviewVisible ? "1" : "0"); }, [layoutRestored, isPreviewVisible]);

  function toggleTheme() {
    setTheme(current => {
      const next: Theme = current === "dark" ? "light" : "dark";
      try {
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("docuchat-theme", next);
      } catch { /* private browsing or storage disabled -- theme still applies for this session */ }
      return next;
    });
  }

  function beginResize(which: "sidebar" | "chat", event: ReactPointerEvent) {
    event.preventDefault();
    let lastX = event.clientX;
    function onMove(moveEvent: PointerEvent) {
      const delta = moveEvent.clientX - lastX;
      lastX = moveEvent.clientX;
      if (which === "sidebar") setSidebarWidth(w => Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, w + delta)));
      else setChatWidth(w => Math.min(CHAT_MAX, Math.max(CHAT_MIN, w - delta)));
    }
    function onUp() {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  const refreshCollections = useCallback(async () => {
    try {
      const items = await listCollections();
      setCollections(items);
      return items;
    } catch { return null; }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [items, status] = await Promise.all([listDocuments(), getSystemStatus()]);
      setApiUnreachable(false);
      setDocuments(items);
      // Only documents that exist AND are ready can be chat context: one that
      // was deleted, or failed while selected, must not stay ticked.
      setSelected(current => pruneSelection(current, items));
      setOpenDocs(current => current.filter(id => items.some(item => item.id === id)));
      setActiveDocId(current => (current && items.some(item => item.id === current) ? current : null));
      setSystem(status);
      await refreshCollections();
    } catch { setApiUnreachable(true); }
    finally { setInitialLoading(false); }
  }, [refreshCollections]);
  useEffect(() => { void refresh(); }, [refresh]);
  // Keep the library current: quickly while a document is processing (so it
  // flips to Ready/Failed the moment the backend finishes), slowly while the
  // backend is unreachable (so a restart is picked up without a page reload).
  const pollMs = refreshInterval(documents, apiUnreachable);
  useEffect(() => {
    if (pollMs === null) return;
    const interval = setInterval(() => { void refresh(); }, pollMs);
    return () => clearInterval(interval);
  }, [pollMs, refresh]);
  // Coming back to a tab that sat in the background (laptop asleep, backend
  // restarted meanwhile) must not show yesterday's state.
  useEffect(() => {
    const onVisible = () => { if (document.visibilityState === "visible") void refresh(); };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => { document.removeEventListener("visibilitychange", onVisible); window.removeEventListener("focus", onVisible); };
  }, [refresh]);

  // Suggested questions are scoped to whichever single document the chat
  // would actually retrieve against: the one checkbox-selected document, or
  // (if none is selected) whichever one is open in the viewer. More than
  // one selected document has no single set of suggestions to show, so the
  // empty state just falls back to the generic prompts instead.
  const suggestionTargetId = (() => {
    const targetId = selected.length === 1 ? selected[0] : selected.length === 0 ? activeDocId : null;
    const targetDoc = targetId ? documents.find(doc => doc.id === targetId) : null;
    return targetDoc && isChattable(targetDoc) ? targetDoc.id : null;
  })();
  useEffect(() => {
    setSuggestedQuestions([]);
    if (!suggestionTargetId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    // A document turns Ready a few seconds before its suggested questions are
    // written, so the first answer may be "pending" -- ask again shortly.
    const load = (attempt: number) => {
      getSuggestedQuestions(suggestionTargetId)
        .then(result => {
          if (cancelled) return;
          setSuggestedQuestions(result.questions);
          const delay = nextSuggestionPoll(attempt, result.pending);
          if (delay !== null) timer = setTimeout(() => load(attempt + 1), delay);
        })
        .catch(() => { if (!cancelled) setSuggestedQuestions([]); });
    };
    load(0);
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [suggestionTargetId]);

  // Per-page OCR status for the document in the viewer (only once it is
  // ready -- a document still processing has no page rows yet).
  const activeDocReady = documents.some(doc => doc.id === activeDocId && doc.status === "ready");
  useEffect(() => {
    if (!activeDocId || !activeDocReady) { setPageInfos([]); return; }
    let cancelled = false;
    getDocumentPages(activeDocId)
      .then(result => { if (!cancelled) setPageInfos(result.pages); })
      .catch(() => { if (!cancelled) setPageInfos([]); });
    return () => { cancelled = true; };
  }, [activeDocId, activeDocReady]);

  async function handleUpload(file: File) {
    setUploading(true); setNotice(null);
    try { await uploadDocument(file); await refresh(); }
    catch (err) { setNotice(uploadNotice(err)); }  // shown by the upload button, not in the chat
    finally { setUploading(false); }
  }
  const selectionDisabled = Boolean(activeCollectionId && selected.length === 0);
  // Group 6: learning modes study "this document" -- the checked documents,
  // or else the one open in the viewer (the same fallback Abstract uses).
  const learningDocumentIds = selected.length > 0 ? selected : activeDocId ? [activeDocId] : [];
  const learningCollectionId = activeCollectionId && selected.length > 0 ? activeCollectionId : undefined;

  async function handleAsk(event: FormEvent) {
    event.preventDefault(); if (!question.trim() || loading || selectionDisabled) return;
    if (mode === "tutor") { await runTutorTurn({ typedText: question }); return; }
    const prompt = question.trim(); const replyingTo = replyTarget; setQuestion(""); setError(""); setLoading(true);
    setMessages(current => [...current, { role: "user", content: prompt }]);
    try {
      const result = mode === "brainstorm"
        ? await askQuestion(prompt, learningDocumentIds, conversationId, {
          collectionId: learningCollectionId, groundingMode, replyToMessageId: replyingTo?.messageId, mode: "brainstorm",
        })
        : await askQuestion(prompt, selected, conversationId, {
          collectionId: activeCollectionId ?? undefined,
          groundingMode,
          replyToMessageId: replyingTo?.messageId,
        });
      setConversationId(result.conversation_id);
      setMessages(current => [...current, assistantMessage(result)]);
      setReplyTarget(null);
    } catch (err) { reportFailure(err, "Could not answer that question."); }
    finally { setLoading(false); }
  }
  // Group 6 TUTOR MODE: typed messages and action chips both become one
  // structured /chat request (see lib/tutor.ts::buildTutorRequest).
  async function runTutorTurn(input: { action?: TutorActionId; typedText?: string }) {
    if (loading || selectionDisabled) return;
    const tutorRequest = buildTutorRequest(
      tutorSession,
      { documentIds: learningDocumentIds, collectionId: learningCollectionId, groundingMode, replyToMessageId: replyTarget?.messageId },
      input,
    );
    if (!tutorRequest.question) return;
    setQuestion(""); setError(""); setLoading(true);
    setMessages(current => [...current, { role: "user", content: tutorRequest.displayText }]);
    try {
      const result = await askQuestion(tutorRequest.question, tutorRequest.documentIds, conversationId, tutorRequest.options);
      setConversationId(result.conversation_id);
      setMessages(current => [...current, assistantMessage(result)]);
      setReplyTarget(null);
      setTutorSession(current => tutorSessionAfter({ ...current, topic: tutorRequest.options.currentTopic ?? current.topic }, result));
    } catch (err) { reportFailure(err, "The tutor couldn't answer that."); }
    finally { setLoading(false); }
  }
  // Group 7 PROMPT TEMPLATES: applying one is a /chat request in "template"
  // mode over the same document/collection scope and grounding mode as Chat.
  async function handleApplyTemplate(template: TemplateItem, focus: string) {
    if (loading || selectionDisabled || learningDocumentIds.length === 0) return;
    const { question: prompt, templateId } = buildTemplateRequest(template, focus);
    setError(""); setLoading(true);
    setMessages(current => [...current, { role: "user", content: prompt }]);
    try {
      const result = await askQuestion(prompt, learningDocumentIds, conversationId, {
        collectionId: learningCollectionId, groundingMode, replyToMessageId: replyTarget?.messageId, mode: "template", templateId,
      });
      setConversationId(result.conversation_id);
      setMessages(current => [...current, assistantMessage(result)]);
      setReplyTarget(null);
    } catch (err) { reportFailure(err, "Could not apply that template."); }
    finally { setLoading(false); }
  }
  // Group 6 SELECTION INTEGRATION: the PDF selection menu's "Tutor" item
  // pins the passage (with its page and document) as the tutor's context;
  // every tutor action then works on it until it's cleared.
  function handleTutorSelection(payload: TutorSelectionPayload) {
    if (!activeDocId) return;
    setTutorSession({ topic: null, pendingQuestion: null, selection: { text: payload.text, page: payload.page, documentId: activeDocId } });
    setMode("tutor");
    setChatCollapsed(false);
    setMobileView("chat");
  }
  async function handleSelectionAction(payload: SelectionPayload) {
    if (loading || !activeDocId) return;
    const actionLabel = payload.action.charAt(0).toUpperCase() + payload.action.slice(1);
    const preview = payload.text.length > 240 ? `${payload.text.slice(0, 240)}…` : payload.text;
    setError(""); setLoading(true);
    setMessages(current => [...current, { role: "user", content: `${actionLabel} selected passage:\n"${preview}"` }]);
    try {
      const result = await askSelectionAction(payload.action, payload.text, [activeDocId], payload.page, conversationId);
      setConversationId(result.conversation_id);
      setMessages(current => [...current, assistantMessage(result)]);
    } catch (err) { reportFailure(err, "Could not process the selected passage."); }
    finally { setLoading(false); }
  }
  // Group 5 quote/source reuse: shares askSelectionAction (Group 4's
  // selection-AI pipeline) instead of a parallel implementation. Forcing
  // document_ids to exactly [citation.document_id] is what guarantees the
  // reused source keeps its original document identity even inside a
  // multi-document collection answer (see CROSS-DOCUMENT SOURCE REUSE).
  async function handleQuoteAction({ citation, action }: QuoteAction) {
    if (loading) return;
    const actionLabel = action.charAt(0).toUpperCase() + action.slice(1);
    setError(""); setLoading(true);
    setMessages(current => [...current, { role: "user", content: `${actionLabel} cited passage from ${citation.filename} (p. ${citation.page_number}):\n"${citation.excerpt}"` }]);
    try {
      const result = await askSelectionAction(action, citation.excerpt, [citation.document_id], citation.page_number, conversationId);
      setConversationId(result.conversation_id);
      setMessages(current => [...current, { role: "assistant", content: result.answer, citations: result.citations, id: result.message_id }]);
    } catch (err) { reportFailure(err, "Could not process the selected source."); }
    finally { setLoading(false); }
  }
  async function handleAbstract() {
    const documentIds = selected.length > 0 ? selected : activeDocId ? [activeDocId] : [];
    if (documentIds.length === 0 || abstractLoading) return;
    setError(""); setAbstractLoading(true);
    setMessages(current => [...current, { role: "user", content: "Generate an abstract of this document." }]);
    try {
      const result = await generateAbstract(documentIds, conversationId);
      setConversationId(result.conversation_id);
      setMessages(current => [...current, { role: "assistant", content: result.answer, citations: result.citations, id: result.message_id }]);
    } catch (err) { reportFailure(err, "Could not generate an abstract for this document."); }
    finally { setAbstractLoading(false); }
  }
  function handleReply(message: Message) {
    if (!message.id) return;
    const preview = message.content.length > 90 ? `${message.content.slice(0, 90)}…` : message.content;
    setReplyTarget({ messageId: message.id, preview });
  }
  function clearReply() { setReplyTarget(null); }
  const chattableIds = new Set(documents.filter(isChattable).map(doc => doc.id));
  function toggleSelect(id: string) {
    if (!chattableIds.has(id)) return; // a document that isn't Ready has nothing to chat with
    // A plain "My Documents" checkbox always means an ad-hoc, non-collection
    // selection -- leaving collection mode here keeps "which documents am I
    // chatting with" unambiguous (see DOCUMENT / COLLECTION CONTEXT).
    setActiveCollectionId(null);
    setSelected(current => current.includes(id) ? current.filter(value => value !== id) : [...current, id]);
  }
  function selectCollection(collectionId: string) {
    const collection = collections.find(item => item.id === collectionId);
    if (!collection) return;
    setActiveCollectionId(collectionId);
    // Every member that can actually be chatted with -- a member still
    // processing (or failed) is listed in the collection but not selectable.
    setSelected(collection.documents.filter(doc => chattableIds.has(doc.id)).map(doc => doc.id));
  }
  function toggleCollectionDocumentSelect(_collectionId: string, documentId: string) {
    if (!chattableIds.has(documentId)) return;
    setSelected(current => current.includes(documentId) ? current.filter(value => value !== documentId) : [...current, documentId]);
  }
  async function handleCreateCollection(name: string) {
    try { await createCollection(name); await refreshCollections(); }
    catch (err) { reportFailure(err, "Could not create the collection."); }
  }
  async function handleRenameCollection(id: string, name: string) {
    try { await renameCollection(id, name); await refreshCollections(); }
    catch (err) { reportFailure(err, "Could not rename the collection."); }
  }
  async function handleDeleteCollection(id: string) {
    try {
      await deleteCollection(id);
      if (activeCollectionId === id) { setActiveCollectionId(null); setSelected([]); }
      await refreshCollections();
    } catch (err) { reportFailure(err, "Could not delete the collection."); }
  }
  async function handleAddDocumentToCollection(collectionId: string, documentId: string) {
    try {
      await addDocumentToCollection(collectionId, documentId);
      await refreshCollections();
      if (activeCollectionId === collectionId && chattableIds.has(documentId)) setSelected(current => (current.includes(documentId) ? current : [...current, documentId]));
    } catch (err) { reportFailure(err, "Could not add the document to that collection."); }
  }
  async function handleRemoveDocumentFromCollection(collectionId: string, documentId: string) {
    try {
      await removeDocumentFromCollection(collectionId, documentId);
      await refreshCollections();
      if (activeCollectionId === collectionId) setSelected(current => current.filter(value => value !== documentId));
    } catch (err) { reportFailure(err, "Could not remove the document from that collection."); }
  }
  async function remove(id: string) {
    try {
      await deleteDocument(id);
      setTutorSession(current => (current.selection?.documentId === id ? { ...current, selection: null } : current));
      setOpenDocs(current => current.filter(docId => docId !== id));
      setActiveDocId(current => (current === id ? null : current));
      await refresh();
    } catch (err) { reportFailure(err, "Could not remove document."); }
  }
  function reportFailure(err: unknown, fallback: string) {
    const report = failureReport(err, fallback);
    if (report.unreachable) setApiUnreachable(true); // shared state: status card, retry poll, clears itself on recovery
    else setError(report.message);
  }
  function newChat() { setMessages([]); setConversationId(undefined); setError(""); setReplyTarget(null); setTutorSession(EMPTY_TUTOR_SESSION); }

  function openDocument(doc: DocumentItem) {
    if (doc.status === "failed") return;
    setOpenDocs(current => (current.includes(doc.id) ? current : [...current, doc.id]));
    setActiveDocId(doc.id);
    setJumpTarget(current => jumpForDocument(current, activeDocId, doc.id)); // openCitation sets its own right after
    setMobileView("document");
  }
  function closeTab(id: string) {
    if (id === activeDocId) setJumpTarget(null); // the next tab's viewer must not inherit this document's jump
    setOpenDocs(current => {
      const next = current.filter(docId => docId !== id);
      setActiveDocId(activeId => (activeId === id ? (next.length ? next[next.length - 1] : null) : activeId));
      return next;
    });
  }
  function openCitation(citation: Citation) {
    const doc = documents.find(item => item.id === citation.document_id);
    if (!doc) { setError("That document is no longer in your library."); return; }
    openDocument(doc);
    setIsPreviewVisible(true);
    setJumpTarget({ page: citation.page_number, bbox: citation.bbox ?? null, nonce: Date.now(), kind: "citation" });
  }
  function navigateToTocPage(page: number) {
    setJumpTarget({ page, bbox: null, nonce: Date.now(), kind: "citation" });
  }
  // Group 7: jump to a figure's page (highlighting its box when known).
  function navigateToFigurePage(page: number, bbox?: BBox | null) {
    setJumpTarget({ page, bbox: bbox ?? null, nonce: Date.now(), kind: "citation" });
  }
  function toggleToc() { setFiguresOpen(false); setTocOpen(current => !current); }
  function toggleFigures() { setTocOpen(false); setFiguresOpen(current => !current); }
  function togglePreview() {
    setIsPreviewVisible(current => {
      const next = !current;
      if (!next) { setTocOpen(false); setFiguresOpen(false); } // no point browsing a hidden document's panels
      return next;
    });
  }

  const activeDoc = documents.find(doc => doc.id === activeDocId) ?? null;
  const currentPageChip = pageChip(pageInfos.find(page => page.page_number === currentPage));
  const activeCollection = collections.find(item => item.id === activeCollectionId) ?? null;
  const sidebarColumn = sidebarCollapsed ? "60px" : `${sidebarWidth}px`;
  const chatColumn = chatCollapsed ? "60px" : `${chatWidth}px`;
  const gridTemplateColumns = isPreviewVisible
    ? `${sidebarColumn} 6px minmax(360px, 1fr) 6px ${chatColumn}`
    : `${sidebarColumn} 6px 0px 0px ${chatCollapsed ? "60px" : "1fr"}`;
  // Group 5 DOCUMENT / COLLECTION CONTEXT: make the active chat scope
  // unambiguous so a user can't mistake "one document" for "a whole
  // collection" or vice versa.
  const contextLabel = activeCollection
    ? `${activeCollection.name} · ${selected.length ? `${selected.length} document${selected.length > 1 ? "s" : ""} selected` : "No documents selected"}`
    : selected.length
      ? `${selected.length} source${selected.length > 1 ? "s" : ""} selected`
      : "No sources selected";
  // Group 6: what the Exam panel will generate questions from.
  const learningScope: LearningScope | null = selectionDisabled || documents.length === 0
    ? null
    : { documentIds: learningDocumentIds, collectionId: learningCollectionId, groundingMode };
  const learningScopeLabel = activeCollection
    ? `${activeCollection.name} (${selected.length} document${selected.length === 1 ? "" : "s"})`
    : learningDocumentIds.length === 1
      ? documents.find(doc => doc.id === learningDocumentIds[0])?.filename ?? "1 document"
      : learningDocumentIds.length > 1 ? `${learningDocumentIds.length} documents` : "all documents in your library";

  return (
    <main className="app-shell" data-mobile-view={mobileView}>
      <div className="workspace-shell" style={{ gridTemplateColumns }}>
        <Sidebar
          documents={documents}
          selected={selected}
          openDocumentId={activeDocId}
          uploading={uploading}
          loading={initialLoading}
          unreachable={apiUnreachable}
          system={systemView(system, apiUnreachable)}
          notice={notice}
          onDismissNotice={() => setNotice(null)}
          onRetry={() => void refresh()}
          theme={theme}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(current => !current)}
          onToggleTheme={toggleTheme}
          onUpload={handleUpload}
          onToggleSelect={toggleSelect}
          onOpenDocument={openDocument}
          onRemove={remove}
          onNewChat={newChat}
          collections={collections}
          activeCollectionId={activeCollectionId}
          onSelectCollection={selectCollection}
          onToggleCollectionDocument={toggleCollectionDocumentSelect}
          onCreateCollection={handleCreateCollection}
          onRenameCollection={handleRenameCollection}
          onDeleteCollection={handleDeleteCollection}
          onAddDocumentToCollection={handleAddDocumentToCollection}
          onRemoveDocumentFromCollection={handleRemoveDocumentFromCollection}
        />
        <div className="pane-resizer" onPointerDown={event => beginResize("sidebar", event)} />

        <section className={`center-pane ${isPreviewVisible ? "" : "is-hidden"}`}>
          <div className="center-topbar">
            <div className="doc-tabs">
              {openDocs.length === 0 && <span className="doc-tabs-empty">No document open</span>}
              {openDocs.map(id => {
                const doc = documents.find(item => item.id === id);
                if (!doc) return null;
                return (
                  <button type="button" key={id} className={`doc-tab ${activeDocId === id ? "is-active" : ""}`} onClick={() => setActiveDocId(id)} title={doc.filename}>
                    <FileIcon size={13} /> <span>{doc.filename}</span>
                    <span className="doc-tab-close" role="button" tabIndex={-1} onClick={event => { event.stopPropagation(); closeTab(id); }} aria-label={`Close ${doc.filename}`}><CloseIcon size={11} /></span>
                  </button>
                );
              })}
            </div>
            <div className="topbar-actions">
              {activeDoc && isPreviewVisible && currentPageChip && (
                <span className={`page-status-chip is-${currentPageChip.tone}`} title={currentPageChip.title} data-testid="page-status-chip">{currentPageChip.label}</span>
              )}
              {activeDoc && isPreviewVisible && (
                <button
                  type="button"
                  className={`topbar-toggle ${tocOpen ? "is-active" : ""}`}
                  onClick={toggleToc}
                  disabled={activeDoc.status === "processing"}
                  aria-label="Toggle table of contents"
                  title={activeDoc.status === "processing" ? "Available once processing finishes" : "Table of contents"}
                >
                  <ListIcon size={15} /> Contents
                </button>
              )}
              {activeDoc && isPreviewVisible && activeDoc.status === "ready" && (
                <button type="button" className={`topbar-toggle ${figuresOpen ? "is-active" : ""}`} onClick={toggleFigures} aria-label="Toggle figures and page previews" title="Figures & page previews">
                  <ImageIcon size={15} /> Figures
                </button>
              )}
              <span className="source-count">{contextLabel}</span>
              <span className="private-badge"><i /> Local only</span>
            </div>
          </div>
          <div className="center-body">
            {activeDoc ? (
              <>
                <PdfViewer
                  key={activeDoc.id}
                  documentId={activeDoc.id}
                  fileUrl={pdfFileUrl(activeDoc.id)}
                  downloadUrl={pdfDownloadUrl(activeDoc.id)}
                  jumpTarget={jumpTarget}
                  onPageChange={setCurrentPage}
                  onSelectionAction={handleSelectionAction}
                  onTutorSelection={handleTutorSelection}
                />
                {tocOpen && (
                  <TocPanel key={`toc-${activeDoc.id}`} documentId={activeDoc.id} currentPage={currentPage} onNavigate={navigateToTocPage} onClose={() => setTocOpen(false)} />
                )}
                {figuresOpen && (
                  <FiguresPanel key={`figures-${activeDoc.id}`} documentId={activeDoc.id} currentPage={currentPage} onNavigate={navigateToFigurePage} onClose={() => setFiguresOpen(false)} />
                )}
              </>
            ) : (
              <div className="center-empty">
                {initialLoading ? (
                  <div className="center-empty-skeleton" />
                ) : (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element -- small fixed brand mark */}
                    <div className="hero-logo"><img src="/docuchat-ai-logo.png" alt="" /></div>
                    <h1>DocuChat <span>AI</span></h1>
                    <p>{documents.length === 0 ? "Upload a PDF from the left to get started." : "Open a document from your library to preview it here."}</p>
                  </>
                )}
              </div>
            )}
          </div>
        </section>

        <div className="pane-resizer" onPointerDown={event => beginResize("chat", event)} />
        <ChatPanel
          messages={messages}
          loading={loading}
          error={error || (apiUnreachable ? UNREACHABLE_MESSAGE : "")}
          question={question}
          onQuestionChange={setQuestion}
          onAsk={handleAsk}
          selectedCount={selected.length}
          selectionDisabled={selectionDisabled}
          collapsed={chatCollapsed}
          onToggleCollapse={() => setChatCollapsed(current => !current)}
          onNewChat={newChat}
          isPreviewVisible={isPreviewVisible}
          onTogglePreview={togglePreview}
          hasActiveDocument={Boolean(activeDoc)}
          onOpenCitation={openCitation}
          onQuoteAction={handleQuoteAction}
          suggestedQuestions={suggestedQuestions}
          onAbstract={handleAbstract}
          abstractLoading={abstractLoading}
          groundingMode={groundingMode}
          onGroundingModeChange={setGroundingMode}
          replyTarget={replyTarget}
          onReply={handleReply}
          onClearReply={clearReply}
          mode={mode}
          onModeChange={setMode}
          tutorControls={(
            <TutorControls
              state={tutorSession}
              typedText={question}
              busy={loading}
              disabled={selectionDisabled}
              onAction={action => void runTutorTurn({ action, typedText: question })}
              onClearTopic={() => setTutorSession(current => ({ ...current, topic: null, pendingQuestion: null }))}
              onClearSelection={() => setTutorSession(current => ({ ...current, selection: null }))}
            />
          )}
          templatePanel={(
            <TemplatePanel
              scopeLabel={learningScopeLabel}
              busy={loading}
              disabledReason={selectionDisabled
                ? "No documents selected in this collection."
                : learningDocumentIds.length === 0 ? "Select or open a document first — a template needs a document to work from." : null}
              onApply={(template, focus) => void handleApplyTemplate(template, focus)}
            />
          )}
          examPanel={(
            <ExamPanel
              state={study}
              dispatch={dispatchStudy}
              scope={learningScope}
              scopeLabel={learningScopeLabel}
              selection={tutorSession.selection}
              onOpenCitation={openCitation}
            />
          )}
        />
      </div>

      <nav className="mobile-tabbar">
        <button type="button" className={mobileView === "library" ? "is-active" : ""} onClick={() => setMobileView("library")}>Library</button>
        <button type="button" className={mobileView === "document" ? "is-active" : ""} onClick={() => setMobileView("document")}>Document</button>
        <button type="button" className={mobileView === "chat" ? "is-active" : ""} onClick={() => setMobileView("chat")}>Assistant</button>
      </nav>
    </main>
  );
}
