"use client";

import { useEffect, useMemo, useState } from "react";
import { getDocumentToc, TocItem } from "../../lib/api";
import { ChevronDownIcon, ChevronRightIcon, CloseIcon, ListIcon } from "./icons";

type TocNode = TocItem & { children: TocNode[] };

/** Turns the flat, level-annotated list the API returns into a real tree,
 * using a stack of the currently-open ancestors at each level -- the same
 * approach a table-of-contents renderer needs regardless of whether the
 * levels came from native PDF bookmarks or the heuristic font-size fallback.
 */
function buildTree(items: TocItem[]): TocNode[] {
  const roots: TocNode[] = [];
  const stack: TocNode[] = [];
  for (const item of items) {
    const node: TocNode = { ...item, children: [] };
    while (stack.length && stack[stack.length - 1].level >= node.level) stack.pop();
    if (stack.length === 0) roots.push(node);
    else stack[stack.length - 1].children.push(node);
    stack.push(node);
  }
  return roots;
}

/** The active item is the last one whose page is <= the page currently
 * shown in the viewer -- i.e. "which section is the user reading right now."
 */
function findActiveId(items: TocItem[], currentPage: number): string | null {
  let activeId: string | null = null;
  for (const item of items) {
    if (item.page <= currentPage) activeId = item.id;
    else break;
  }
  return activeId;
}

function TocNodeRow({ node, activeId, onNavigate }: { node: TocNode; activeId: string | null; onNavigate: (page: number) => void }) {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = node.children.length > 0;
  return (
    <li className="toc-node">
      <div className={`toc-row ${activeId === node.id ? "is-active" : ""}`} style={{ paddingLeft: 10 + (node.level - 1) * 16 }}>
        {hasChildren ? (
          <button type="button" className="toc-expand" onClick={() => setExpanded(current => !current)} aria-label={expanded ? "Collapse section" : "Expand section"}>
            {expanded ? <ChevronDownIcon size={12} /> : <ChevronRightIcon size={12} />}
          </button>
        ) : (
          <span className="toc-expand-spacer" />
        )}
        <button type="button" className="toc-title" onClick={() => onNavigate(node.page)} title={node.title}>
          <span>{node.title}</span>
          <span className="toc-page">{node.page}</span>
        </button>
      </div>
      {hasChildren && expanded && (
        <ul>
          {node.children.map(child => <TocNodeRow key={child.id} node={child} activeId={activeId} onNavigate={onNavigate} />)}
        </ul>
      )}
    </li>
  );
}

type Props = {
  documentId: string;
  currentPage: number;
  onNavigate: (page: number) => void;
  onClose: () => void;
};

export default function TocPanel({ documentId, currentPage, onNavigate, onClose }: Props) {
  const [items, setItems] = useState<TocItem[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setItems(null);
    setError("");
    getDocumentToc(documentId)
      .then(result => { if (!cancelled) setItems(result.items); })
      .catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : "Could not load the table of contents."); });
    return () => { cancelled = true; };
  }, [documentId]);

  const tree = useMemo(() => (items ? buildTree(items) : []), [items]);
  const activeId = useMemo(() => (items ? findActiveId(items, currentPage) : null), [items, currentPage]);

  return (
    <div className="toc-panel">
      <header className="toc-header">
        <span className="toc-heading"><ListIcon size={15} /> Contents</span>
        <button type="button" className="toc-close" onClick={onClose} aria-label="Close table of contents"><CloseIcon size={13} /></button>
      </header>
      <div className="toc-body">
        {error && <div className="toc-state toc-error">{error}</div>}
        {!error && items === null && (
          <div className="toc-skeleton">
            <div /><div /><div /><div />
          </div>
        )}
        {!error && items !== null && items.length === 0 && (
          <div className="toc-state toc-empty">No table of contents was found for this document.</div>
        )}
        {!error && tree.length > 0 && (
          <ul className="toc-tree">
            {tree.map(node => <TocNodeRow key={node.id} node={node} activeId={activeId} onNavigate={onNavigate} />)}
          </ul>
        )}
      </div>
    </div>
  );
}
