"use client";

import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { BBox, SearchMatch, SelectionAction, searchDocument } from "../../lib/api";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  DownloadIcon,
  FitWidthIcon,
  FullscreenExitIcon,
  FullscreenIcon,
  SearchIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from "./icons";

// The worker does the actual PDF parsing off the main thread; without this
// react-pdf silently fails to render anything. `new URL(..., import.meta.url)`
// is the pattern Next.js's webpack build resolves into a static asset URL.
pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

export type JumpTarget = { page: number; bbox: BBox | null; nonce: number; kind?: "citation" | "search" };
export type SelectionPayload = { text: string; page: number; bbox: BBox | null; action: SelectionAction };
// Group 6: a selection handed to the tutor (pinned as its context) rather
// than run through one of the Group 4 one-shot actions above.
export type TutorSelectionPayload = { text: string; page: number; bbox: BBox | null };

type SelectionMenuState = { x: number; y: number; placeAbove: boolean; text: string; page: number; bbox: BBox | null };

type Props = {
  documentId: string;
  fileUrl: string;
  downloadUrl: string;
  jumpTarget: JumpTarget | null;
  onPageChange?: (page: number) => void;
  onSelectionAction?: (payload: SelectionPayload) => void;
  onTutorSelection?: (payload: TutorSelectionPayload) => void;
};

const SELECTION_ACTIONS: { action: SelectionAction; label: string }[] = [
  { action: "explain", label: "Explain" },
  { action: "summarize", label: "Summarize" },
  { action: "analyze", label: "Analyze" },
  { action: "rewrite", label: "Rewrite" },
];
const SELECTION_MENU_MARGIN = 8;
const SELECTION_MENU_ESTIMATED_WIDTH = 330; // 4 Group 4 actions + the Group 6 "Tutor" item
const SELECTION_MENU_ESTIMATED_HEIGHT = 44;

const MIN_SCALE = 0.4;
const MAX_SCALE = 3;
const ZOOM_STEP = 0.15;
// Citation highlights fade after a while so they don't linger forever; an
// active search match stays lit the whole time the user is stepping through
// results, since it's clearly tied to something they're still doing.
const CITATION_HIGHLIGHT_DURATION_MS = 4500;

function PdfSkeleton() {
  return <div className="pdf-skeleton"><div className="pdf-skeleton-line" style={{ width: "70%" }} /><div className="pdf-skeleton-line" style={{ width: "94%" }} /><div className="pdf-skeleton-line" style={{ width: "88%" }} /><div className="pdf-skeleton-line" style={{ width: "60%" }} /></div>;
}

export default function PdfViewer({ documentId, fileUrl, downloadUrl, jumpTarget, onPageChange, onSelectionAction, onTutorSelection }: Props) {
  const [numPages, setNumPages] = useState(0);
  const [pageNumber, setPageNumber] = useState(1);
  const [pageInput, setPageInput] = useState("1");
  const [fitMode, setFitMode] = useState<"width" | "custom">("width");
  const [scale, setScale] = useState(1);
  const [pageSize, setPageSize] = useState<{ width: number; height: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [activeHighlight, setActiveHighlight] = useState<JumpTarget | null>(null);
  const [selectionMenu, setSelectionMenu] = useState<SelectionMenuState | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);

  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchMatches, setSearchMatches] = useState<SearchMatch[] | null>(null);
  const [searchActiveIndex, setSearchActiveIndex] = useState(0);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");
  const searchRequestId = useRef(0);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setNumPages(0);
    setPageNumber(1);
    setPageInput("1");
    setPageSize(null);
    setLoadError("");
    setActiveHighlight(null);
    setSearchOpen(false);
    setSearchQuery("");
    setSearchMatches(null);
    setSearchActiveIndex(0);
    setSearchError("");
    setSelectionMenu(null);
  }, [fileUrl]);

  // Selection-based AI (Group 4): detect a real text selection made inside
  // this document's canvas area (the pdf.js text layer, enabled below via
  // renderTextLayer) and show a small floating action menu near it. Scoped
  // to selectionchange rather than mouseup so it also clears correctly when
  // a selection is dismissed by clicking elsewhere or pressing Escape.
  useEffect(() => {
    function onSelectionChange() {
      const selection = window.getSelection();
      const container = containerRef.current;
      if (!selection || selection.isCollapsed || !container) { setSelectionMenu(null); return; }
      const anchorNode = selection.anchorNode;
      if (!anchorNode || !container.contains(anchorNode)) { setSelectionMenu(null); return; }
      const text = selection.toString().trim();
      if (!text) { setSelectionMenu(null); return; }

      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) { setSelectionMenu(null); return; }

      const canvas = container.querySelector(".react-pdf__Page__canvas") as HTMLCanvasElement | null;
      let bbox: BBox | null = null;
      if (canvas) {
        const canvasRect = canvas.getBoundingClientRect();
        bbox = [
          Math.round(((rect.left - canvasRect.left) / scale) * 100) / 100,
          Math.round(((rect.top - canvasRect.top) / scale) * 100) / 100,
          Math.round(((rect.right - canvasRect.left) / scale) * 100) / 100,
          Math.round(((rect.bottom - canvasRect.top) / scale) * 100) / 100,
        ];
      }

      const placeAbove = rect.top > SELECTION_MENU_ESTIMATED_HEIGHT + SELECTION_MENU_MARGIN * 2;
      const x = Math.min(
        Math.max(SELECTION_MENU_MARGIN, rect.left + rect.width / 2 - SELECTION_MENU_ESTIMATED_WIDTH / 2),
        window.innerWidth - SELECTION_MENU_ESTIMATED_WIDTH - SELECTION_MENU_MARGIN,
      );
      const y = placeAbove ? rect.top - SELECTION_MENU_ESTIMATED_HEIGHT - SELECTION_MENU_MARGIN : Math.min(rect.bottom + SELECTION_MENU_MARGIN, window.innerHeight - SELECTION_MENU_ESTIMATED_HEIGHT - SELECTION_MENU_MARGIN);

      setSelectionMenu({ x, y, placeAbove, text, page: pageNumber, bbox });
    }
    document.addEventListener("selectionchange", onSelectionChange);
    return () => document.removeEventListener("selectionchange", onSelectionChange);
  }, [pageNumber, scale]);

  function runSelectionAction(action: SelectionAction) {
    if (!selectionMenu) return;
    onSelectionAction?.({ text: selectionMenu.text, page: selectionMenu.page, bbox: selectionMenu.bbox, action });
    window.getSelection()?.removeAllRanges();
    setSelectionMenu(null);
  }
  function sendSelectionToTutor() {
    if (!selectionMenu) return;
    onTutorSelection?.({ text: selectionMenu.text, page: selectionMenu.page, bbox: selectionMenu.bbox });
    window.getSelection()?.removeAllRanges();
    setSelectionMenu(null);
  }

  useEffect(() => {
    onPageChange?.(pageNumber);
  }, [pageNumber, onPageChange]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(entries => {
      const width = entries[0]?.contentRect.width;
      if (width) setContainerWidth(width);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (fitMode === "width" && pageSize && containerWidth) {
      setScale(Math.min(MAX_SCALE, Math.max(MIN_SCALE, (containerWidth - 48) / pageSize.width)));
    }
  }, [fitMode, pageSize, containerWidth]);

  useEffect(() => {
    if (!jumpTarget) return;
    setPageNumber(current => Math.min(Math.max(1, jumpTarget.page), numPages || jumpTarget.page || current));
    setPageInput(String(jumpTarget.page));
    setActiveHighlight(jumpTarget);
    if (jumpTarget.kind !== "search") {
      const timeout = setTimeout(() => setActiveHighlight(current => (current?.nonce === jumpTarget.nonce ? null : current)), CITATION_HIGHLIGHT_DURATION_MS);
      return () => clearTimeout(timeout);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only the citation click (nonce) should retrigger this, not every numPages recalculation
  }, [jumpTarget]);

  useEffect(() => {
    function onFullscreenChange() { setIsFullscreen(Boolean(document.fullscreenElement)); }
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  function goToPage(next: number) {
    const clamped = Math.min(Math.max(1, next), numPages || next);
    setPageNumber(clamped);
    setPageInput(String(clamped));
    setSelectionMenu(null);
  }
  function submitPageInput(event: React.FormEvent) {
    event.preventDefault();
    const parsed = Number.parseInt(pageInput, 10);
    if (Number.isFinite(parsed)) goToPage(parsed);
    else setPageInput(String(pageNumber));
  }
  function zoomIn() { setFitMode("custom"); setScale(current => Math.min(MAX_SCALE, Math.round((current + ZOOM_STEP) * 100) / 100)); }
  function zoomOut() { setFitMode("custom"); setScale(current => Math.max(MIN_SCALE, Math.round((current - ZOOM_STEP) * 100) / 100)); }
  function toggleFullscreen() {
    if (!viewerRef.current) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void viewerRef.current.requestFullscreen();
  }

  function goToMatch(index: number) {
    if (!searchMatches || searchMatches.length === 0) return;
    const clamped = ((index % searchMatches.length) + searchMatches.length) % searchMatches.length;
    setSearchActiveIndex(clamped);
    const match = searchMatches[clamped];
    setPageNumber(current => Math.min(Math.max(1, match.page), numPages || match.page || current));
    setPageInput(String(match.page));
    setActiveHighlight({ page: match.page, bbox: match.bbox, nonce: Date.now(), kind: "search" });
  }

  async function submitSearch(event: React.FormEvent) {
    event.preventDefault();
    const query = searchQuery.trim();
    if (!query) { setSearchMatches(null); setSearchError(""); return; }
    const requestId = ++searchRequestId.current;
    setSearchLoading(true);
    setSearchError("");
    try {
      const result = await searchDocument(documentId, query);
      if (requestId !== searchRequestId.current) return; // a newer search superseded this one
      setSearchMatches(result.matches);
      setSearchActiveIndex(0);
      if (result.matches.length > 0) {
        const first = result.matches[0];
        setPageNumber(current => Math.min(Math.max(1, first.page), numPages || first.page || current));
        setPageInput(String(first.page));
        setActiveHighlight({ page: first.page, bbox: first.bbox, nonce: Date.now(), kind: "search" });
      } else {
        setActiveHighlight(current => (current?.kind === "search" ? null : current));
      }
    } catch (err) {
      if (requestId !== searchRequestId.current) return;
      setSearchMatches(null);
      setSearchError(err instanceof Error ? err.message : "Search failed.");
    } finally {
      if (requestId === searchRequestId.current) setSearchLoading(false);
    }
  }

  function clearSearch() {
    searchRequestId.current += 1; // invalidate any in-flight request
    setSearchQuery("");
    setSearchMatches(null);
    setSearchActiveIndex(0);
    setSearchError("");
    setSearchLoading(false);
    setActiveHighlight(current => (current?.kind === "search" ? null : current));
  }

  function toggleSearch() {
    setSearchOpen(current => {
      const next = !current;
      if (!next) clearSearch();
      return next;
    });
  }

  useEffect(() => {
    if (searchOpen) searchInputRef.current?.focus();
  }, [searchOpen]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const inField = Boolean(target && (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable));
      if (event.key === "Escape" && searchOpen) { event.preventDefault(); toggleSearch(); return; }
      if (event.key === "Escape" && selectionMenu) { event.preventDefault(); window.getSelection()?.removeAllRanges(); setSelectionMenu(null); return; }
      if (inField) return;
      // These are single-key shortcuts. With a modifier held the key belongs to
      // the browser or OS (Cmd/Ctrl+F is "find", Cmd/Ctrl+S "save", Cmd/Ctrl+-
      // "zoom out", Alt+Left "back") -- hijacking them, as this once did, breaks
      // the page around the viewer. Ctrl+= is the one deliberate exception.
      const zoomChord = event.ctrlKey && !event.metaKey && !event.altKey && event.key === "=";
      if ((event.metaKey || event.ctrlKey || event.altKey) && !zoomChord) return;
      if (event.key === "ArrowRight") { event.preventDefault(); goToPage(pageNumber + 1); }
      else if (event.key === "ArrowLeft") { event.preventDefault(); goToPage(pageNumber - 1); }
      else if (event.key === "+" || (event.ctrlKey && event.key === "=")) { event.preventDefault(); zoomIn(); }
      else if (event.key === "-") { event.preventDefault(); zoomOut(); }
      else if (event.key.toLowerCase() === "f") { event.preventDefault(); toggleFullscreen(); }
      else if (event.key.toLowerCase() === "s") { event.preventDefault(); setSearchOpen(true); }
      else if (event.key === "Home") { event.preventDefault(); goToPage(1); }
      else if (event.key === "End" && numPages) { event.preventDefault(); goToPage(numPages); }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- handlers close over state already covered by pageNumber/numPages/searchOpen/selectionMenu
  }, [pageNumber, numPages, searchOpen, selectionMenu]);

  const highlightStyle = useMemo(() => {
    if (!activeHighlight || !activeHighlight.bbox || !pageSize || activeHighlight.page !== pageNumber) return null;
    const [x0, y0, x1, y1] = activeHighlight.bbox;
    return { left: x0 * scale, top: y0 * scale, width: Math.max(3, (x1 - x0) * scale), height: Math.max(3, (y1 - y0) * scale) };
  }, [activeHighlight, pageSize, pageNumber, scale]);

  useEffect(() => {
    if (!highlightStyle) return;
    const frame = requestAnimationFrame(() => {
      highlightRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    return () => cancelAnimationFrame(frame);
  }, [highlightStyle]);

  const totalMatches = searchMatches?.length ?? 0;

  return (
    <div className={`pdf-viewer ${isFullscreen ? "is-fullscreen" : ""}`} ref={viewerRef}>
      <div className="pdf-toolbar">
        <div className="pdf-toolbar-group">
          <button type="button" onClick={() => goToPage(pageNumber - 1)} disabled={pageNumber <= 1} title="Previous page (←)" aria-label="Previous page"><ChevronLeftIcon /></button>
          <form className="pdf-page-jump" onSubmit={submitPageInput}>
            <input value={pageInput} onChange={event => setPageInput(event.target.value)} onFocus={event => event.target.select()} inputMode="numeric" aria-label="Page number" />
            <span>/ {numPages || "–"}</span>
          </form>
          <button type="button" onClick={() => goToPage(pageNumber + 1)} disabled={numPages > 0 && pageNumber >= numPages} title="Next page (→)" aria-label="Next page"><ChevronRightIcon /></button>
        </div>
        <div className="pdf-toolbar-group">
          <button type="button" onClick={zoomOut} title="Zoom out (-)" aria-label="Zoom out"><ZoomOutIcon /></button>
          <span className="pdf-zoom-level">{Math.round(scale * 100)}%</span>
          <button type="button" onClick={zoomIn} title="Zoom in (+)" aria-label="Zoom in"><ZoomInIcon /></button>
          <button type="button" className={fitMode === "width" ? "is-active" : ""} onClick={() => setFitMode("width")} title="Fit to width" aria-label="Fit to width"><FitWidthIcon /></button>
        </div>
        <div className="pdf-toolbar-group">
          <button type="button" className={searchOpen ? "is-active" : ""} onClick={toggleSearch} title="Search in document (S)" aria-label="Search in document"><SearchIcon /></button>
          <a className="pdf-icon-button" href={downloadUrl} title="Download" aria-label="Download PDF"><DownloadIcon /></a>
          <button type="button" onClick={toggleFullscreen} title="Fullscreen (F)" aria-label="Toggle fullscreen">{isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}</button>
        </div>
      </div>

      {searchOpen && (
        <form className="pdf-search-bar" onSubmit={submitSearch}>
          <SearchIcon size={15} className="pdf-search-icon" />
          <input
            ref={searchInputRef}
            value={searchQuery}
            onChange={event => setSearchQuery(event.target.value)}
            placeholder="Search in this document…"
            aria-label="Search text in document"
          />
          {searchLoading && <span className="pdf-search-spinner" aria-hidden="true" />}
          {!searchLoading && searchMatches !== null && (
            <span className="pdf-search-count">{totalMatches > 0 ? `${searchActiveIndex + 1} / ${totalMatches}` : "No matches"}</span>
          )}
          <button type="button" onClick={() => goToMatch(searchActiveIndex - 1)} disabled={totalMatches === 0} title="Previous match" aria-label="Previous match"><ChevronLeftIcon size={16} /></button>
          <button type="button" onClick={() => goToMatch(searchActiveIndex + 1)} disabled={totalMatches === 0} title="Next match" aria-label="Next match"><ChevronRightIcon size={16} /></button>
          <button type="button" onClick={clearSearch} title="Clear search" aria-label="Clear search"><CloseIcon size={14} /></button>
        </form>
      )}
      {searchOpen && searchError && <div className="pdf-search-error">{searchError}</div>}

      <div className="pdf-canvas-area" ref={containerRef}>
        {loadError ? (
          <div className="pdf-error">{loadError}</div>
        ) : (
          <Document
            file={fileUrl}
            // react-pdf defaults to Suspense-based loading, which needs a
            // Suspense boundary scoped to just this tree. Without one, a
            // page change suspends up into next/dynamic's own boundary
            // (used to load this component client-only) and remounts this
            // entire viewer -- wiping page/zoom state on every navigation.
            // suspense={false} reverts to the classic loading/error/
            // onLoadSuccess callback API used below, which is what this
            // component is built around.
            suspense={false}
            onLoadSuccess={({ numPages: total }) => { setNumPages(total); setLoadError(""); }}
            onLoadError={() => setLoadError("This PDF could not be displayed.")}
            loading={<PdfSkeleton />}
            error={<div className="pdf-error">This PDF could not be displayed.</div>}
          >
            <div className="pdf-page-wrap" style={pageSize ? { width: pageSize.width * scale, height: pageSize.height * scale } : undefined}>
              <Page
                pageNumber={pageNumber}
                scale={scale}
                suspense={false}
                onLoadSuccess={page => { const viewport = page.getViewport({ scale: 1 }); setPageSize({ width: viewport.width, height: viewport.height }); }}
                renderTextLayer
                renderAnnotationLayer={false}
                loading={<PdfSkeleton />}
              />
              {highlightStyle && (
                <div
                  ref={highlightRef}
                  className={`pdf-highlight ${activeHighlight?.kind === "search" ? "pdf-highlight-search" : ""}`}
                  style={highlightStyle}
                />
              )}
            </div>
          </Document>
        )}
      </div>

      {selectionMenu && (
        <div className="selection-menu" style={{ left: selectionMenu.x, top: selectionMenu.y }} role="menu" aria-label="Selection actions">
          {SELECTION_ACTIONS.map(({ action, label }) => (
            <button type="button" key={action} onClick={() => runSelectionAction(action)} role="menuitem">{label}</button>
          ))}
          {onTutorSelection && (
            <button type="button" className="selection-menu-tutor" onClick={sendSelectionToTutor} role="menuitem" title="Pin this passage in Tutor mode">Tutor</button>
          )}
        </div>
      )}
    </div>
  );
}
