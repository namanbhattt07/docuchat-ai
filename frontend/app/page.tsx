"use client";

import { ChangeEvent, FormEvent, ReactNode, useEffect, useRef, useState } from "react";
import { askQuestion, Citation, deleteDocument, DocumentItem, getSystemStatus, listDocuments, uploadDocument } from "../lib/api";

type Message = { role: "user" | "assistant"; content: string; citations?: Citation[] };

const SUGGESTED_PROMPTS = [
  "Give me a summary of this document",
  "What are the key topics covered?",
  "What is the main idea of this document?",
];

function FormattedText({ text }: { text: string }) {
  return <>{text.split("\n").map((line, lineIndex) => <p key={lineIndex}>{line.split(/(\*\*[^*]+\*\*)/g).map((part, partIndex): ReactNode => part.startsWith("**") && part.endsWith("**") ? <strong key={partIndex}>{part.slice(2, -2)}</strong> : part)}</p>)}</>;
}

function statusLabel(doc: DocumentItem): string {
  if (doc.status === "ready") return `Ready · ${doc.page_count} pages`;
  if (doc.status === "processing") return "Processing…";
  if (doc.status === "empty") return "No text found";
  return "Failed";
}

function statusClass(doc: DocumentItem): string {
  if (doc.status === "ready") return "doc-ready";
  if (doc.status === "failed") return "doc-failed";
  if (doc.status === "empty") return "doc-warning";
  return "";
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [conversationId, setConversationId] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [ollamaReady, setOllamaReady] = useState<boolean | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  async function refresh() {
    try {
      const [items, system] = await Promise.all([listDocuments(), getSystemStatus()]);
      setDocuments(items);
      setSelected(current => current.filter(id => items.some(item => item.id === id)));
      setOllamaReady(system.ollama.available && system.ollama.generation_model_ready && system.ollama.embedding_model_ready);
    } catch { setError("Could not connect to the DocuChat API. Is the backend running?"); }
  }
  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    // Ingestion now runs in the background, so poll while anything is still
    // processing instead of blocking the upload request on it.
    if (!documents.some(doc => doc.status === "processing")) return;
    const interval = setInterval(() => { void refresh(); }, 2000);
    return () => clearInterval(interval);
  }, [documents]);

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; if (!file) return;
    setUploading(true); setError("");
    try { await uploadDocument(file); await refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : "Upload failed."); }
    finally { setUploading(false); event.target.value = ""; }
  }
  async function handleAsk(event: FormEvent) {
    event.preventDefault(); if (!question.trim() || loading) return;
    const prompt = question.trim(); setQuestion(""); setError(""); setLoading(true);
    setMessages(current => [...current, { role: "user", content: prompt }]);
    try {
      const result = await askQuestion(prompt, selected, conversationId);
      setConversationId(result.conversation_id);
      setMessages(current => [...current, { role: "assistant", content: result.answer, citations: result.citations }]);
    } catch (err) { setError(err instanceof Error ? err.message : "Could not answer that question."); }
    finally { setLoading(false); }
  }
  function toggle(id: string) { setSelected(current => current.includes(id) ? current.filter(value => value !== id) : [...current, id]); }
  async function remove(id: string) { try { await deleteDocument(id); await refresh(); } catch (err) { setError(err instanceof Error ? err.message : "Could not remove document."); } }
  function newChat() { setMessages([]); setConversationId(undefined); setError(""); }
  function selectPrompt(prompt: string) { setQuestion(prompt); composerRef.current?.focus(); }

  return <main className="app-shell">
    <aside className="sidebar">
      <div className="sidebar-top"><div className="brand"><span className="brand-orbit"><i>D</i></span><span>DocuChat <b>AI</b></span></div><button className="new-chat" onClick={newChat}><span>＋</span> New conversation</button></div>
      <div className="library-heading"><span>Your library</span><b>{documents.length}</b></div>
      <button className="upload-zone" onClick={() => inputRef.current?.click()} disabled={uploading}><span className="upload-symbol">↥</span><span><strong>{uploading ? "Uploading…" : "Add a document"}</strong><small>PDF · stored locally</small></span><i>＋</i></button>
      <input ref={inputRef} className="hidden" type="file" accept="application/pdf,.pdf" onChange={handleUpload} />
      <div className="document-list">
        {documents.length === 0 && <div className="empty-library"><span>◌</span><p>Your knowledge base<br />is ready for its first PDF.</p></div>}
        {documents.map(doc => <article className={`document-card ${selected.includes(doc.id) ? "active" : ""}`} key={doc.id}>
          <label className="document-select"><input type="checkbox" checked={selected.includes(doc.id)} onChange={() => toggle(doc.id)} /><span className="custom-check">✓</span><span className="file-badge">PDF</span><span className="document-name">{doc.filename}</span></label>
          <footer><span className={statusClass(doc)} title={doc.status_detail ?? undefined}>{statusLabel(doc)}</span><button onClick={() => void remove(doc.id)} aria-label={`Remove ${doc.filename}`}>×</button></footer>
        </article>)}
      </div>
      <div className={`system-card ${ollamaReady ? "ready" : ""}`}><span className="system-pulse" /><div><strong>{ollamaReady ? "Private AI is ready" : "Model setup needed"}</strong><small>{ollamaReady ? "Running locally on this Mac" : "Connect Ollama to begin"}</small></div></div>
    </aside>

    <section className="workspace">
      <header className="topbar"><div><p className="eyebrow">DOCUMENT WORKSPACE</p><h1>{messages.length ? "Your research thread" : "Ask your documents anything."}</h1></div><div className="topbar-actions"><span className="source-count">{selected.length ? `${selected.length} source${selected.length > 1 ? "s" : ""} selected` : "No sources selected"}</span><span className="private-badge"><i /> Local only</span></div></header>
      <div className={`conversation ${messages.length === 0 ? "is-empty" : ""}`}>
        {messages.length === 0 && <section className="hero">
          {/* eslint-disable-next-line @next/next/no-img-element -- absolute-positioned crop technique in globals.css needs a plain img, not next/image's fixed box */}
          <div className="hero-logo"><img src="/docuchat-ai-logo.png" alt="" /></div>
          <h2>DocuChat <span>AI</span></h2>
          <p className="hero-copy">Upload a PDF and ask anything — every answer is grounded in your document, with page citations.</p>
          <div className="prompt-grid">
            {SUGGESTED_PROMPTS.map(prompt => <button type="button" key={prompt} onClick={() => selectPrompt(prompt)}><span>{prompt}</span><em>→</em></button>)}
          </div>
        </section>}
        {messages.map((message, index) => <article className={`message ${message.role}`} key={index}><i>{message.role === "assistant" ? "✦" : "N"}</i><div className="message-content"><FormattedText text={message.content} />{message.citations && message.citations.length > 0 && <section className="citations"><p className="evidence-label">EVIDENCE USED</p>{message.citations.map(cite => <details key={cite.index}><summary><b>[{cite.index}]</b> {cite.filename} <span>Page {cite.page_number}</span></summary><p>{cite.excerpt}</p></details>)}</section>}</div></article>)}
        {loading && <article className="message assistant"><i>✦</i><div className="thinking"><span /><span /><span /> Searching your selected evidence</div></article>}
      </div>
      {error && <div className="error">{error}</div>}
      <form className="composer" onSubmit={handleAsk}><span className="composer-spark">✦</span><textarea ref={composerRef} value={question} onChange={event => setQuestion(event.target.value)} placeholder={selected.length ? "Ask a question about selected documents…" : "Select a document, then ask a question…"} rows={1} /><button type="submit" disabled={!question.trim() || loading} aria-label="Send question">↑</button></form>
      <small className="hint">Responses are generated locally. Sources shown below an answer are the only evidence it used.</small>
    </section>
  </main>;
}
