"use client";

import { ChangeEvent, FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";
import { AssistantMode, BrainstormMeta, Citation, GroundingMode, SelectionAction, TutorMeta, VisualMeta } from "../../lib/api";
import { BookIcon, ChevronRightIcon, CloseIcon, ExternalLinkIcon, EyeIcon, EyeOffIcon, GlobeIcon, LockIcon, PlusIcon, ReplyIcon, SendIcon, SparkleIcon, TemplateIcon } from "./icons";
import { BrainstormAnswer, TutorAnswer } from "./LearningAnswers";
import MarkdownContent from "./MarkdownContent";
import VisualPreview from "./VisualPreview";

// Group 6: tutor/brainstorm replies carry their structured payload alongside
// the Markdown `content`, so they render as labelled sections (see
// LearningAnswers) while plain chat answers render exactly as before.
// Group 7: `visual` (chart/figure questions) and `template` (an applied prompt
// template) ride on the same message shape; both render as part of the normal
// Markdown answer + citations rather than a separate view.
export type Message = {
  role: "user" | "assistant"; content: string; citations?: Citation[]; id?: string; tutor?: TutorMeta; brainstorm?: BrainstormMeta;
  visual?: VisualMeta; template?: { id: string; name: string };
};
export type ReplyTarget = { messageId: string; preview: string };
// Group 5 quote/source reuse: reuses the exact same action set (and request
// pipeline) as Group 4's selection-based AI -- see PdfViewer's
// SELECTION_ACTIONS and askSelectionAction in lib/api.ts.
export type QuoteAction = { citation: Citation; action: SelectionAction };

const SUGGESTED_PROMPTS = ["Give me a summary of this document", "What are the key topics covered?", "What is the main idea of this document?"];
const TEXTAREA_MAX_HEIGHT = 130;
// The backend cuts a citation's excerpt at this many characters without marking
// the cut, so a full-length excerpt is shown with an ellipsis.
const EXCERPT_LIMIT = 280;
const QUOTE_ACTIONS: { action: SelectionAction; label: string }[] = [
  { action: "explain", label: "Explain" },
  { action: "summarize", label: "Summarize" },
  { action: "analyze", label: "Analyze" },
  { action: "rewrite", label: "Rewrite" },
];

function FormattedText({ text }: { text: string }) {
  return <>{text.split("\n").map((line, lineIndex) => <p key={lineIndex}>{line.split(/(\*\*[^*]+\*\*)/g).map((part, partIndex): ReactNode => part.startsWith("**") && part.endsWith("**") ? <strong key={partIndex}>{part.slice(2, -2)}</strong> : part)}</p>)}</>;
}

// Group 6 mode selector + per-mode copy. Exam is a separate panel (no chat
// thread); Tutor and Brainstorm share the thread and composer with Chat.
const MODES: { id: AssistantMode; label: string; title: string }[] = [
  { id: "chat", label: "Chat", title: "Ask questions about your documents" },
  { id: "tutor", label: "Tutor", title: "Learn step by step with a document-grounded tutor" },
  { id: "exam", label: "Exam", title: "Practice exams, question bank and writing feedback" },
  { id: "brainstorm", label: "Brainstorm", title: "Generate ideas, kept separate from document evidence" },
  { id: "template", label: "Templates", title: "Apply a structured prompt template (case brief, study guide, ...) to your document" },
];
const MODE_EMPTY_STATES: Record<Exclude<AssistantMode, "exam" | "chat">, { title: string; copy: string; prompts: string[] }> = {
  template: {
    title: "Apply a template to your document",
    copy: "Choose a template below — a research paper review, case brief, study guide and more. Each section is grounded in your selected document(s) with page citations, and anything the document doesn't cover is marked as such.",
    prompts: [],
  },
  tutor: {
    title: "Learn with your tutor",
    copy: "Ask about a concept or pick an action below. The tutor explains step by step, checks your understanding, and marks what comes from your document versus its own explanation.",
    prompts: ["Teach me the main idea of this document", "What should I understand first in this document?", "Explain the most important concept here"],
  },
  brainstorm: {
    title: "Brainstorm from your document",
    copy: "Ideas are generated from your document's evidence — the evidence is cited, and generated ideas are always labelled separately.",
    prompts: ["Brainstorm project ideas based on this document", "Give me possible applications of the main concept", "Suggest questions I could investigate further"],
  },
};
const LOADING_TEXT: Record<Exclude<AssistantMode, "exam">, string> = {
  chat: "Searching your selected evidence",
  tutor: "Your tutor is preparing an explanation",
  brainstorm: "Gathering evidence and brainstorming",
  template: "Applying the template to your evidence",
};

function AssistantMessage({ message, onOpenCitation, onReply, onQuoteAction }: { message: Message; onOpenCitation: (citation: Citation) => void; onReply?: (message: Message) => void; onQuoteAction: (payload: QuoteAction) => void }) {
  const [visibleChars, setVisibleChars] = useState(0);
  const [quoteMenuFor, setQuoteMenuFor] = useState<number | null>(null);
  const text = message.content;
  const structured = Boolean(message.tutor || message.brainstorm);
  useEffect(() => {
    // Structured (Group 6) replies render as labelled sections, not a
    // character-by-character reveal -- and neither does anyone who asked their
    // system to reduce motion.
    const reduceMotion = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (structured || reduceMotion) { setVisibleChars(text.length); return; }
    setVisibleChars(0);
    if (!text) return;
    const step = Math.max(1, Math.round(text.length / 40));
    const id = setInterval(() => {
      setVisibleChars(current => {
        const next = current + step;
        if (next >= text.length) { clearInterval(id); return text.length; }
        return next;
      });
    }, 20);
    return () => clearInterval(id);
  }, [text, structured]);
  const revealed = visibleChars >= text.length;
  // The "Use this source" menu closes on Escape or a click anywhere else.
  useEffect(() => {
    if (quoteMenuFor === null) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!(event.target as Element | null)?.closest(".quote-reuse")) setQuoteMenuFor(null);
    };
    const onKeyDown = (event: globalThis.KeyboardEvent) => { if (event.key === "Escape") setQuoteMenuFor(null); };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => { document.removeEventListener("mousedown", onPointerDown); document.removeEventListener("keydown", onKeyDown); };
  }, [quoteMenuFor]);
  return (
    <div className="message-content">
      {message.template && <p className="template-badge"><TemplateIcon size={12} /> Template · {message.template.name}</p>}
      {message.tutor ? <TutorAnswer tutor={message.tutor} />
        : message.brainstorm ? <BrainstormAnswer brainstorm={message.brainstorm} />
          : <MarkdownContent text={text.slice(0, visibleChars)} />}
      {revealed && message.visual && <VisualPreview visual={message.visual} onOpenCitation={onOpenCitation} />}
      {revealed && message.citations && message.citations.length > 0 && (
        <section className="citations">
          <p className="evidence-label">Evidence used</p>
          {message.citations.map(cite => (
            <details key={cite.index}>
              <summary>
                <b>[{cite.index}]</b> {cite.filename} <span>Page {cite.page_number}{cite.section ? ` · ${cite.section}` : ""}</span>{" "}
                {cite.source_type === "ocr" && <em className="source-chip is-ocr" title="Recognized from a scanned page by OCR; may contain recognition errors">OCR</em>}
                {cite.source_type === "figure" && <em className="source-chip is-figure" title="A detected figure or image on this page">Figure</em>}
              </summary>
              <p>{cite.excerpt}{cite.excerpt.length >= EXCERPT_LIMIT ? "…" : ""}</p>
              {cite.bbox_source === "unavailable" && (
                <p className="citation-fallback-note">Source passage could not be located precisely — opening the page instead.</p>
              )}
              <div className="citation-actions">
                <button type="button" className="open-in-pdf" onClick={() => onOpenCitation(cite)}><ExternalLinkIcon size={13} /> Open in PDF</button>
                <div className="quote-reuse">
                  <button
                    type="button"
                    className="quote-reuse-toggle"
                    onClick={() => setQuoteMenuFor(current => (current === cite.index ? null : cite.index))}
                    aria-expanded={quoteMenuFor === cite.index}
                  >
                    Use this source
                  </button>
                  {quoteMenuFor === cite.index && (
                    <div className="quote-reuse-menu" role="menu" aria-label="Actions on this source">
                      {QUOTE_ACTIONS.map(({ action, label }) => (
                        <button
                          type="button"
                          key={action}
                          role="menuitem"
                          onClick={() => { setQuoteMenuFor(null); onQuoteAction({ citation: cite, action }); }}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </details>
          ))}
        </section>
      )}
      {revealed && onReply && message.id && (
        <button type="button" className="reply-affordance" onClick={() => onReply(message)}>
          <ReplyIcon size={12} /> Reply
        </button>
      )}
    </div>
  );
}

type Props = {
  messages: Message[];
  loading: boolean;
  error: string;
  question: string;
  onQuestionChange: (value: string) => void;
  onAsk: (event: FormEvent) => void;
  selectedCount: number;
  selectionDisabled?: boolean;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onNewChat: () => void;
  onOpenCitation: (citation: Citation) => void;
  onQuoteAction: (payload: QuoteAction) => void;
  isPreviewVisible: boolean;
  onTogglePreview: () => void;
  hasActiveDocument: boolean;
  suggestedQuestions: string[];
  onAbstract: () => void;
  abstractLoading: boolean;
  groundingMode: GroundingMode;
  onGroundingModeChange: (mode: GroundingMode) => void;
  replyTarget: ReplyTarget | null;
  onReply: (message: Message) => void;
  onClearReply: () => void;
  // Group 6 learning modes. `tutorControls` renders above the composer in
  // Tutor mode; `examPanel` replaces the thread + composer in Exam mode.
  mode: AssistantMode;
  onModeChange: (mode: AssistantMode) => void;
  tutorControls?: ReactNode;
  examPanel?: ReactNode;
  // Group 7: replaces the free-text composer in Templates mode.
  templatePanel?: ReactNode;
};

export default function ChatPanel({
  messages, loading, error, question, onQuestionChange, onAsk, selectedCount, selectionDisabled, collapsed, onToggleCollapse,
  onNewChat, onOpenCitation, onQuoteAction, isPreviewVisible, onTogglePreview, hasActiveDocument, suggestedQuestions, onAbstract,
  abstractLoading, groundingMode, onGroundingModeChange, replyTarget, onReply, onClearReply, mode, onModeChange, tutorControls, examPanel, templatePanel,
}: Props) {
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading, abstractLoading]);

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter while an IME (Chinese/Japanese/Korean input) is composing confirms
    // the candidate -- it must not also send the half-typed message.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }
  function handleComposerChange(event: ChangeEvent<HTMLTextAreaElement>) {
    onQuestionChange(event.target.value);
    const el = event.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, TEXTAREA_MAX_HEIGHT)}px`;
  }
  function selectPrompt(prompt: string) {
    onQuestionChange(prompt);
    composerRef.current?.focus();
  }
  function handleAsk(event: FormEvent) {
    onAsk(event);
    if (composerRef.current) composerRef.current.style.height = "auto";
  }
  function handleReply(message: Message) {
    onReply(message);
    composerRef.current?.focus();
  }

  if (collapsed) {
    return (
      <aside className="chat-panel is-collapsed">
        <button type="button" className="collapse-toggle" onClick={onToggleCollapse} title="Expand assistant" aria-label="Expand assistant"><ChevronRightIcon /></button>
        <span className="rail-icon-static"><SparkleIcon size={16} /></span>
      </aside>
    );
  }

  const composerDisabled = Boolean(selectionDisabled);
  // Kept short enough to fit the one-row composer at the narrowest chat width.
  const composerPlaceholder = composerDisabled
    ? "No documents selected."
    : mode === "tutor"
      ? "Ask the tutor, or answer its question…"
      : mode === "brainstorm"
        ? "What should we brainstorm?"
        : selectedCount
          ? "Ask about the selected documents…"
          : "Ask about your documents…";
  const emptyState = mode === "tutor" || mode === "brainstorm" || mode === "template" ? MODE_EMPTY_STATES[mode] : null;

  return (
    <aside className="chat-panel">
      <header className="chat-header">
        <span className="chat-title"><span className="chat-icon"><SparkleIcon size={15} /></span>AI Assistant</span>
        <div className="chat-header-actions">
          {hasActiveDocument && (
            <button
              type="button"
              className="abstract-button"
              onClick={onAbstract}
              disabled={abstractLoading}
              aria-label="Generate a structured abstract of this document"
              title="Generate a structured abstract of this document"
            >
              <BookIcon size={13} />
              <span>{abstractLoading ? "Generating…" : "Abstract"}</span>
            </button>
          )}
          {hasActiveDocument && (
            <button
              type="button"
              className="preview-toggle"
              onClick={onTogglePreview}
              aria-label={isPreviewVisible ? "Hide PDF preview" : "Show PDF preview"}
              title={isPreviewVisible ? "Hide Preview" : "Show Preview"}
            >
              {isPreviewVisible ? <EyeOffIcon size={14} /> : <EyeIcon size={14} />}
              <span>{isPreviewVisible ? "Hide Preview" : "Show Preview"}</span>
            </button>
          )}
          <button type="button" onClick={onNewChat} title="New conversation" aria-label="New conversation"><PlusIcon size={16} /></button>
          <button type="button" className="collapse-toggle" onClick={onToggleCollapse} title="Collapse assistant" aria-label="Collapse assistant"><ChevronRightIcon /></button>
        </div>
      </header>

      <div className="mode-selector" role="tablist" aria-label="Assistant mode">
        {MODES.map(item => (
          <button
            type="button"
            key={item.id}
            role="tab"
            aria-selected={mode === item.id}
            className={mode === item.id ? "is-active" : ""}
            onClick={() => onModeChange(item.id)}
            title={item.title}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="grounding-toggle" role="radiogroup" aria-label="Grounding mode">
        <button
          type="button"
          role="radio"
          aria-checked={groundingMode === "document"}
          className={groundingMode === "document" ? "is-active" : ""}
          onClick={() => onGroundingModeChange("document")}
          title="Answer strictly from the selected documents; say so plainly when they don't cover it."
        >
          <LockIcon size={12} /> Stick to Document
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={groundingMode === "free"}
          className={groundingMode === "free" ? "is-active" : ""}
          onClick={() => onGroundingModeChange("free")}
          title="Use document evidence first, then allow clearly-labelled general knowledge to fill gaps."
        >
          <GlobeIcon size={12} /> Go Freely
        </button>
      </div>

      {mode === "exam" ? <>{error && <div className="error" role="alert">{error}</div>}{examPanel}</> : (<>
      <div className="chat-thread" ref={threadRef} role="log" aria-label="Conversation" aria-live="polite">
        {messages.length === 0 && emptyState && (
          <div className="chat-empty">
            <p className="chat-empty-title">{emptyState.title}</p>
            <p className="chat-empty-copy">{emptyState.copy}</p>
            <div className="prompt-grid">
              {emptyState.prompts.map(prompt => <button type="button" key={prompt} onClick={() => selectPrompt(prompt)}><span>{prompt}</span><span className="prompt-arrow">→</span></button>)}
            </div>
          </div>
        )}
        {messages.length === 0 && !emptyState && (
          <div className="chat-empty">
            {/* eslint-disable-next-line @next/next/no-img-element -- small fixed brand mark */}
            <img className="chat-empty-logo" src="/docuchat-ai-logo.png" alt="" />
            <p className="chat-empty-title">Ask your documents anything</p>
            <p className="chat-empty-copy">Every answer is grounded in your PDFs, with page citations you can open directly in the viewer.</p>
            {suggestedQuestions.length > 0 && <p className="evidence-label suggested-label">Suggested questions</p>}
            <div className="prompt-grid">
              {(suggestedQuestions.length > 0 ? suggestedQuestions : SUGGESTED_PROMPTS).map(prompt => <button type="button" key={prompt} onClick={() => selectPrompt(prompt)}><span>{prompt}</span><span className="prompt-arrow">→</span></button>)}
            </div>
          </div>
        )}
        {messages.map((message, index) => (
          <article className={`message ${message.role}`} key={index}>
            <i>{message.role === "assistant" ? <SparkleIcon size={13} /> : "N"}</i>
            {message.role === "assistant"
              ? <AssistantMessage message={message} onOpenCitation={onOpenCitation} onReply={handleReply} onQuoteAction={onQuoteAction} />
              : <div className="message-content"><FormattedText text={message.content} /></div>}
          </article>
        ))}
        {loading && <article className="message assistant"><i><SparkleIcon size={13} /></i><div className="thinking"><span /><span /><span /> {LOADING_TEXT[mode]}</div></article>}
        {abstractLoading && <article className="message assistant"><i><SparkleIcon size={13} /></i><div className="thinking"><span /><span /><span /> Generating document abstract</div></article>}
      </div>

      {error && <div className="error" role="alert">{error}</div>}

      {replyTarget && (
        <div className="reply-target">
          <ReplyIcon size={12} />
          <span>Replying to: {replyTarget.preview}</span>
          <button type="button" onClick={onClearReply} aria-label="Cancel reply"><CloseIcon size={11} /></button>
        </div>
      )}

      {mode === "tutor" && tutorControls}

      {mode === "template" ? templatePanel : (<>
      <form className="composer" onSubmit={handleAsk}>
        <textarea ref={composerRef} value={question} onChange={handleComposerChange} onKeyDown={handleComposerKeyDown} placeholder={composerPlaceholder} rows={1} disabled={composerDisabled} />
        <button type="submit" disabled={!question.trim() || loading || composerDisabled} aria-label="Send question"><SendIcon /></button>
      </form>
      <small className="hint">Enter to send · Shift+Enter for a new line</small>
      </>)}
      </>)}
    </aside>
  );
}
