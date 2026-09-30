"use client";

import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import { CollectionItem, DocumentItem } from "../../lib/api";
import { isChattable, selectionBlockedReason, SystemView, UploadNotice } from "../../lib/documents";
import { documentStatusView, StatusTone } from "../../lib/ocr";
import { ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, CloseIcon, FileIcon, FolderIcon, MoonIcon, PencilIcon, PlusIcon, SunIcon, TrashIcon } from "./icons";

type Theme = "light" | "dark";

type Props = {
  documents: DocumentItem[];
  selected: string[];
  openDocumentId: string | null;
  uploading: boolean;
  // Group 8: distinguish "still loading" and "backend unreachable" from "no
  // documents yet", and say what is wrong with the local AI when it is.
  loading: boolean;
  unreachable: boolean;
  system: SystemView;
  notice: UploadNotice | null;
  onDismissNotice: () => void;
  onRetry: () => void;
  theme: Theme;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onToggleTheme: () => void;
  onUpload: (file: File) => void;
  onToggleSelect: (id: string) => void;
  onOpenDocument: (doc: DocumentItem) => void;
  onRemove: (id: string) => void;
  onNewChat: () => void;
  // Group 5: multi-document collections.
  collections: CollectionItem[];
  activeCollectionId: string | null;
  onSelectCollection: (collectionId: string) => void;
  onToggleCollectionDocument: (collectionId: string, documentId: string) => void;
  onCreateCollection: (name: string) => void;
  onRenameCollection: (collectionId: string, name: string) => void;
  onDeleteCollection: (collectionId: string) => void;
  onAddDocumentToCollection: (collectionId: string, documentId: string) => void;
  onRemoveDocumentFromCollection: (collectionId: string, documentId: string) => void;
};

const BADGE_VARIANTS = ["badge-a", "badge-b", "badge-c"];

// Group 7: the label/note come from lib/ocr.ts so ingestion stages ("OCR
// processing page 2 (1 of 4)"), OCR failures and blank documents are worded
// (and told apart) in one tested place.
const TONE_CLASS: Record<StatusTone, string> = { ready: "doc-ready", processing: "doc-processing", warning: "doc-warning", failed: "doc-failed" };

function DocumentStatus({ doc }: { doc: DocumentItem }) {
  const view = documentStatusView(doc);
  return (
    <>
      <span className={TONE_CLASS[view.tone]} data-testid="document-status" title={view.title ?? undefined}>{doc.status === "processing" && <i className="status-spinner" />}{view.label}{view.badge && <em className="source-chip is-ocr">{view.badge}</em>}</span>
      {view.note && <span className={`document-note ${TONE_CLASS[view.noteTone ?? "warning"]}`} title={view.note} data-testid="document-status-note">{view.note}</span>}
    </>
  );
}

export default function Sidebar({
  documents, selected, openDocumentId, uploading, loading, unreachable, system, notice, onDismissNotice, onRetry, theme, collapsed, onToggleCollapse, onToggleTheme, onUpload,
  onToggleSelect, onOpenDocument, onRemove, onNewChat, collections, activeCollectionId, onSelectCollection,
  onToggleCollectionDocument, onCreateCollection, onRenameCollection, onDeleteCollection, onAddDocumentToCollection,
  onRemoveDocumentFromCollection,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [creatingCollection, setCreatingCollection] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [addMenuFor, setAddMenuFor] = useState<string | null>(null);

  // The "add to collection" popover closes on Escape or a click anywhere else.
  useEffect(() => {
    if (!addMenuFor) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!(event.target as Element | null)?.closest(".document-collection-menu")) setAddMenuFor(null);
    };
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") setAddMenuFor(null); };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => { document.removeEventListener("mousedown", onPointerDown); document.removeEventListener("keydown", onKeyDown); };
  }, [addMenuFor]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) onUpload(file);
    event.target.value = "";
  }

  function toggleExpanded(id: string) {
    setExpandedIds(current => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  function selectCollection(collection: CollectionItem) {
    setExpandedIds(current => new Set(current).add(collection.id));
    onSelectCollection(collection.id);
  }

  function submitNewCollection(event: FormEvent) {
    event.preventDefault();
    const name = newCollectionName.trim();
    if (!name) return;
    onCreateCollection(name);
    setNewCollectionName("");
    setCreatingCollection(false);
  }

  function startRename(collection: CollectionItem) {
    setRenamingId(collection.id);
    setRenameValue(collection.name);
  }
  function submitRename(event: FormEvent) {
    event.preventDefault();
    const name = renameValue.trim();
    if (name && renamingId) onRenameCollection(renamingId, name);
    setRenamingId(null);
  }

  if (collapsed) {
    return (
      <aside className="sidebar is-collapsed">
        <button type="button" className="collapse-toggle" onClick={onToggleCollapse} title="Expand sidebar" aria-label="Expand sidebar"><ChevronRightIcon /></button>
        {/* eslint-disable-next-line @next/next/no-img-element -- small fixed brand mark, not a next/image candidate */}
        <img className="brand-mark-rail" src="/docuchat-ai-logo.png" alt="DocuChat AI" />
        <button type="button" className="rail-icon-button" onClick={() => inputRef.current?.click()} disabled={uploading} title="Add a document" aria-label="Add a document"><PlusIcon size={16} /></button>
        <input ref={inputRef} className="hidden" type="file" accept="application/pdf,.pdf" onChange={handleFileChange} />
        <div className="rail-doc-list">
          {documents.map(doc => (
            <button type="button" key={doc.id} className={`rail-doc ${openDocumentId === doc.id ? "is-active" : ""}`} onClick={() => onOpenDocument(doc)} title={doc.filename}>
              <FileIcon size={14} />
            </button>
          ))}
        </div>
        <span className={`rail-status-dot ${system.ready ? "is-ready" : ""}`} title={`${system.headline} — ${system.detail}`} />
      </aside>
    );
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <div className="brand">
          {/* eslint-disable-next-line @next/next/no-img-element -- small fixed brand mark */}
          <img className="brand-mark" src="/docuchat-ai-logo.png" alt="" />
          <span>DocuChat <b>AI</b></span>
          <button type="button" className="theme-toggle" onClick={onToggleTheme} aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>{theme === "dark" ? <SunIcon /> : <MoonIcon />}</button>
          <button type="button" className="collapse-toggle" onClick={onToggleCollapse} title="Collapse sidebar" aria-label="Collapse sidebar"><ChevronLeftIcon /></button>
        </div>
        <button type="button" className="upload-zone" onClick={() => inputRef.current?.click()} disabled={uploading}>
          <PlusIcon size={16} />
          <span>{uploading ? "Uploading…" : "Upload Document"}</span>
        </button>
        <input ref={inputRef} className="hidden" type="file" accept="application/pdf,.pdf" onChange={handleFileChange} />
        {notice && (
          <div className={`upload-notice is-${notice.tone}`} role={notice.tone === "error" ? "alert" : "status"} data-testid="upload-notice">
            <span>{notice.text}</span>
            <button type="button" onClick={onDismissNotice} aria-label="Dismiss message"><CloseIcon size={11} /></button>
          </div>
        )}
        <button type="button" className="new-chat" onClick={onNewChat}><PlusIcon size={14} /> New conversation</button>
      </div>

      <div className="library-heading collections-heading">
        <span>Collections</span>
        <button type="button" className="collections-add" onClick={() => setCreatingCollection(current => !current)} title="New collection" aria-label="New collection"><PlusIcon size={13} /></button>
      </div>
      {creatingCollection && (
        <form className="new-collection-form" onSubmit={submitNewCollection}>
          <input autoFocus value={newCollectionName} onChange={event => setNewCollectionName(event.target.value)} placeholder="Collection name" maxLength={200} />
          <button type="submit" disabled={!newCollectionName.trim()}>Create</button>
          <button type="button" onClick={() => { setCreatingCollection(false); setNewCollectionName(""); }} aria-label="Cancel"><CloseIcon size={12} /></button>
        </form>
      )}
      <div className="collection-list">
        {collections.length === 0 && !creatingCollection && <p className="collection-empty-hint">Group documents into a collection to chat across all of them at once.</p>}
        {collections.map(collection => {
          const expanded = expandedIds.has(collection.id);
          const isActive = activeCollectionId === collection.id;
          return (
            <div className={`collection-group ${isActive ? "active" : ""}`} key={collection.id}>
              <div className="collection-header">
                <button type="button" className="collection-expand" onClick={() => toggleExpanded(collection.id)} aria-label={expanded ? "Collapse collection" : "Expand collection"}>
                  <ChevronDownIcon size={13} className={expanded ? "is-expanded" : ""} />
                </button>
                {renamingId === collection.id ? (
                  <form className="collection-rename-form" onSubmit={submitRename}>
                    <input autoFocus value={renameValue} onChange={event => setRenameValue(event.target.value)} onBlur={submitRename} maxLength={200} />
                  </form>
                ) : (
                  <button type="button" className="collection-name-button" title={collection.name} onClick={() => selectCollection(collection)}>
                    <FolderIcon size={14} /> <span>{collection.name}</span> <b>{collection.documents.length}</b>
                  </button>
                )}
                <div className="collection-actions">
                  <button type="button" onClick={() => startRename(collection)} title="Rename collection" aria-label={`Rename ${collection.name}`}><PencilIcon size={12} /></button>
                  <button type="button" onClick={() => onDeleteCollection(collection.id)} title="Delete collection (keeps documents)" aria-label={`Delete ${collection.name}`}><TrashIcon size={12} /></button>
                </div>
              </div>
              {expanded && (
                <div className="collection-documents">
                  {collection.documents.length === 0 && <p className="collection-empty">No documents yet — add one from the list below.</p>}
                  {collection.documents.map(doc => (
                    <div className="collection-document-row" key={doc.id}>
                      <label className="document-select" title={selectionBlockedReason(doc) ?? (isActive ? "Include in collection chat" : "Open this collection to change selection")}>
                        <input
                          type="checkbox"
                          checked={isActive ? selected.includes(doc.id) : isChattable(doc)}
                          disabled={!isActive || !isChattable(doc)}
                          onChange={() => onToggleCollectionDocument(collection.id, doc.id)}
                        />
                        <span className="custom-check">✓</span>
                      </label>
                      <button
                        type="button"
                        className="collection-document-name"
                        title={doc.filename}
                        onClick={() => {
                          const full = documents.find(item => item.id === doc.id);
                          if (full) onOpenDocument(full);
                        }}
                      >
                        {doc.filename}
                      </button>
                      <button
                        type="button"
                        className="collection-document-remove"
                        onClick={() => onRemoveDocumentFromCollection(collection.id, doc.id)}
                        title="Remove from collection (keeps the document)"
                        aria-label={`Remove ${doc.filename} from ${collection.name}`}
                      >
                        <CloseIcon size={11} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="library-heading"><span>My Documents</span><b>{documents.length}</b></div>
      <div className="document-list" aria-busy={loading}>
        {loading && documents.length === 0 && (
          <>
            {[0, 1, 2].map(index => <div className="document-skeleton" key={index} aria-hidden="true" />)}
          </>
        )}
        {!loading && unreachable && documents.length === 0 && (
          <div className="empty-library is-error" role="alert">
            <p>Can’t reach the DocuChat backend.<br />It will reconnect on its own, or</p>
            <button type="button" className="empty-library-retry" onClick={onRetry}>Try again</button>
          </div>
        )}
        {!loading && !unreachable && documents.length === 0 && !uploading && <div className="empty-library"><FileIcon size={22} /><p>Your knowledge base<br />is ready for its first PDF.</p></div>}
        {documents.map((doc, index) => (
          <article className={`document-card ${openDocumentId === doc.id ? "active" : ""}`} key={doc.id}>
            <button type="button" className="document-open" onClick={() => onOpenDocument(doc)} disabled={doc.status === "failed"} title={doc.filename}>
              <span className={`file-badge ${BADGE_VARIANTS[index % BADGE_VARIANTS.length]}`}><FileIcon size={16} /></span>
              <span className="document-meta">
                <span className="document-name">{doc.filename}</span>
                <DocumentStatus doc={doc} />
              </span>
            </button>
            <div className="document-collection-menu">
              <button type="button" className="document-collection-toggle" onClick={() => setAddMenuFor(current => (current === doc.id ? null : doc.id))} title="Add to collection" aria-label={`Add ${doc.filename} to a collection`}>
                <FolderIcon size={14} />
              </button>
              {addMenuFor === doc.id && (
                <div className="document-collection-popover" role="menu">
                  {collections.length === 0 && <p className="collection-empty-hint">No collections yet.</p>}
                  {collections.map(collection => {
                    const isMember = collection.documents.some(item => item.id === doc.id);
                    return (
                      <label key={collection.id} className="document-collection-option">
                        <input
                          type="checkbox"
                          checked={isMember}
                          onChange={() => (isMember ? onRemoveDocumentFromCollection(collection.id, doc.id) : onAddDocumentToCollection(collection.id, doc.id))}
                        />
                        <span>{collection.name}</span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
            <label className="document-select" title={selectionBlockedReason(doc) ?? "Include in chat context"}>
              <input
                type="checkbox"
                checked={selected.includes(doc.id)}
                disabled={!isChattable(doc)}
                aria-label={`Include ${doc.filename} in chat context`}
                onChange={() => onToggleSelect(doc.id)}
              />
              <span className="custom-check">✓</span>
            </label>
            <button type="button" className="document-remove" onClick={() => onRemove(doc.id)} aria-label={`Remove ${doc.filename}`} title="Remove"><TrashIcon /></button>
          </article>
        ))}
      </div>

      <div className={`system-card ${system.ready ? "ready" : ""}`} role="status">
        <span className="system-pulse" />
        <div><strong>{system.headline}</strong><small title={system.detail}>{system.detail}</small></div>
      </div>
    </aside>
  );
}
