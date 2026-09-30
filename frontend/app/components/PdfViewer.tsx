"use client";

import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { BBox, SearchMatch, SelectionAction, searchDocument } from "../../lib/api";
import {
  effectiveScale, fitWidthScale, initialZoom, pageAtProbe, parseViewMode, toggleFit, VIEW_MODE_STORAGE_KEY, zoomBy, type ViewMode, type ZoomState,
} from "../../lib/pdfView";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  DownloadIcon,
  FitWidthIcon,
  FullscreenExitIcon,
  FullscreenIcon,
  PageScrollIcon,
  PageSingleIcon,
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

// Citation highlights fade after a while so they don't linger forever; an
// active search match stays lit the whole time the user is stepping through
// results, since it's clearly tied to something they're still doing.
const CITATION_HIGHLIGHT_DURATION_MS = 4500;

// Continuous-scroll mode. Every page gets a correctly sized placeholder, but a
// canvas is only drawn for pages within this distance of the viewport, so a
// 200-page document costs a handful of canvases, not 200.
const SCROLL_RENDER_MARGIN = "1200px 0px";
// Where a programmatic jump lands a page: a little air above its top edge.
const SCROLL_PAGE_TOP_OFFSET = 16;
// Letter-size fallback until a page's real size is known.
const FALLBACK_PAGE_SIZE: PageSize = { width: 612, height: 792 };
const PAGE_SIZE_BATCH = 20;
// After a programmatic jump the scroll events it causes are not the reader
// moving, so they must not rewrite the current page for this long.
const JUMP_SETTLE_MS = 600;

type PageSize = { width: number; height: number };
type HighlightBox = { left: number; top: number; width: number; height: number; search: boolean };
// The part of a loaded PDF this viewer needs to learn every page's size (the
// real PDFDocumentProxy satisfies it).
type PdfDocLike = { numPages: number; getPage: (pageNumber: number) => Promise<{ getViewport: (options: { scale: number }) => PageSize }> };

function PdfSkeleton() {
  return <div className="pdf-skeleton"><div className="pdf-skeleton-line" style={{ width: "70%" }} /><div className="pdf-skeleton-line" style={{ width: "94%" }} /><div className="pdf-skeleton-line" style={{ width: "88%" }} /><div className="pdf-skeleton-line" style={{ width: "60%" }} /></div>;
}

function readStoredViewMode(): ViewMode {
  try { return parseViewMode(window.localStorage.getItem(VIEW_MODE_STORAGE_KEY)); } catch { return "single"; }
}

type ScrollPageProps = {
  pageNumber: number;
  scale: number;
  width: number;
  height: number;
  root: HTMLElement | null;
  highlight: HighlightBox | null;
  highlightRef: RefObject<HTMLDivElement | null>;
  registerElement: (pageNumber: number, element: HTMLDivElement | null) => void;
  onSize: (pageNumber: number, size: PageSize) => void;
};

// One page in continuous-scroll mode. The wrapper always exists at the page's
// real size (so scroll positions and page jumps are exact); the pdf.js canvas
// is only mounted while the page is near the viewport.
const ScrollPage = memo(function ScrollPage({ pageNumber, scale, width, height, root, highlight, highlightRef, registerElement, onSize }: ScrollPageProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);

  // A layout effect, so the parent's own layout effects (which scroll to a page)
  // can already find this element.
  useLayoutEffect(() => {
    registerElement(pageNumber, wrapperRef.current);
    return () => registerElement(pageNumber, null);
  }, [pageNumber, registerElement]);

  useEffect(() => {
    const element = wrapperRef.current;
    if (!element || !root || typeof IntersectionObserver === "undefined") { setNear(true); return; }
    const observer = new IntersectionObserver(entries => { setNear(entries[entries.length - 1].isIntersecting); }, { root, rootMargin: SCROLL_RENDER_MARGIN });
    observer.observe(element);
    return () => observer.disconnect();
  }, [root]);

  return (
    <div className="pdf-page-wrap pdf-scroll-page" ref={wrapperRef} data-page={pageNumber} style={{ width, height }}>
      {near && (
        <Page
          pageNumber={pageNumber}
          scale={scale}
          suspense={false}
          onLoadSuccess={page => { const viewport = page.getViewport({ scale: 1 }); onSize(pageNumber, { width: viewport.width, height: viewport.height }); }}
          renderTextLayer
          renderAnnotationLayer={false}
          loading={<PdfSkeleton />}
        />
      )}
      {highlight && (
        <div
          ref={highlightRef}
          className={`pdf-highlight ${highlight.search ? "pdf-highlight-search" : ""}`}
          style={{ left: highlight.left, top: highlight.top, width: highlight.width, height: highlight.height }}
        />
      )}
    </div>
  );
});

export default function PdfViewer({ documentId, fileUrl, downloadUrl, jumpTarget, onPageChange, onSelectionAction, onTutorSelection }: Props) {
  const [numPages, setNumPages] = useState(0);
  const [pageNumber, setPageNumber] = useState(1);
  const [pageInput, setPageInput] = useState("1");
  // Fit to width is a mode, not a zoom level: `zoom.fit` follows the viewer's
  // width while on, `zoom.manual` is the last zoom chosen by hand (see lib/pdfView.ts).
  const [zoom, setZoom] = useState<ZoomState>(initialZoom);
  const [viewMode, setViewMode] = useState<ViewMode>(readStoredViewMode);
  // `pageSize` is the page currently drawn in single-page mode; `pageSizes` holds
  // every page's size (filled in for continuous scrolling, where all pages need a
  // correctly sized placeholder before they are drawn).
  const [pageSize, setPageSize] = useState<PageSize | null>(null);
  const [pageSizes, setPageSizes] = useState<Record<number, PageSize>>({});
  const [containerWidth, setContainerWidth] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [activeHighlight, setActiveHighlight] = useState<JumpTarget | null>(null);
  const [selectionMenu, setSelectionMenu] = useState<SelectionMenuState | null>(null);
  const [areaElement, setAreaElement] = useState<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const highlightRef = useRef<HTMLDivElement | null>(null);
  const pdfRef = useRef<PdfDocLike | null>(null);
  const pageElements = useRef(new Map<number, HTMLDivElement>());
  const pageNumberRef = useRef(1);
  const pageFromScroll = useRef(false);
  const jump = useRef<{ page: number; until: number }>({ page: 1, until: 0 });
  const scrollFrame = useRef(0);
  const sizesLoadedFor = useRef<string | null>(null);
  const sizeLoadToken = useRef(0);

  const setAreaRef = useCallback((node: HTMLDivElement | null) => { containerRef.current = node; setAreaElement(node); }, []);
  const registerPageElement = useCallback((page: number, element: HTMLDivElement | null) => {
    if (element) pageElements.current.set(page, element);
    else pageElements.current.delete(page);
  }, []);

  // In continuous mode the scale is fitted to the first page, since one scale has
  // to serve the whole column; in single-page mode it tracks the page on screen.
  const fitBasis = viewMode === "scroll" ? (pageSizes[1] ?? pageSize) : pageSize;
  const fitScale = fitBasis && containerWidth ? fitWidthScale(containerWidth, fitBasis.width) : null;
  const scale = effectiveScale(zoom, fitScale);
  const sizeOf = useCallback((page: number): PageSize => pageSizes[page] ?? pageSizes[1] ?? pageSize ?? FALLBACK_PAGE_SIZE, [pageSizes, pageSize]);

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
    setPageSizes({});
    pdfRef.current = null;
    sizesLoadedFor.current = null;
    sizeLoadToken.current += 1;
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

      // The page the selection is actually on: in continuous-scroll mode that is
      // not necessarily the page the toolbar says is current.
      const anchorElement = anchorNode instanceof Element ? anchorNode : anchorNode.parentElement;
      const pageElement = anchorElement?.closest<HTMLElement>(".react-pdf__Page") ?? null;
      const selectedPage = Number(pageElement?.dataset.pageNumber) || pageNumberRef.current;
      const canvas = (pageElement ?? container).querySelector<HTMLCanvasElement>(".react-pdf__Page__canvas");
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

      setSelectionMenu({ x, y, placeAbove, text, page: selectedPage, bbox });
    }
    document.addEventListener("selectionchange", onSelectionChange);
    return () => document.removeEventListener("selectionchange", onSelectionChange);
  }, [scale]);

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
    pageNumberRef.current = pageNumber;
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
    if (!jumpTarget) return;
    setPageNumber(current => Math.min(Math.max(1, jumpTarget.page), numPages || jumpTarget.page || current));
    setPageInput(String(jumpTarget.page));
    scrollToPage(Math.min(Math.max(1, jumpTarget.page), numPages || jumpTarget.page)); // continuous mode; covers a jump within the page already current
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

  // ---- continuous-scroll mode ------------------------------------------------
  // Refs mirror state that event handlers and effects need the latest value of.
  const viewModeRef = useRef(viewMode);
  const fitScaleRef = useRef(fitScale);
  useLayoutEffect(() => { viewModeRef.current = viewMode; fitScaleRef.current = fitScale; });

  // Scrolls a page's top edge into view (scroll mode only). Jumps are instant so
  // the page number never flickers through the pages in between, and they open a
  // short window in which the scroll events they cause are not treated as the
  // reader moving (see handleAreaScroll).
  const scrollToPage = useCallback((page: number) => {
    const area = containerRef.current;
    const element = pageElements.current.get(page);
    if (viewModeRef.current !== "scroll" || !area || !element) return;
    jump.current = { page, until: Date.now() + JUMP_SETTLE_MS };
    const top = element.getBoundingClientRect().top - area.getBoundingClientRect().top + area.scrollTop - SCROLL_PAGE_TOP_OFFSET;
    area.scrollTo({ top: Math.max(0, top), behavior: "instant" });
  }, []);

  // Page changes that did not come from scrolling (buttons, the page box, a
  // citation, search, keyboard, or switching into this mode) scroll to that page.
  useLayoutEffect(() => {
    if (viewMode !== "scroll" || numPages === 0) return;
    if (pageFromScroll.current) { pageFromScroll.current = false; return; }
    scrollToPage(pageNumber);
  }, [pageNumber, viewMode, numPages, scrollToPage]);

  // Zooming (or a resize while fitting) reflows every page, so stay on the page being read.
  useLayoutEffect(() => { scrollToPage(pageNumberRef.current); }, [scale, scrollToPage]);

  // Page sizes arriving late shift the placeholders below them; while a jump is
  // still settling, land it again on the corrected layout.
  useLayoutEffect(() => {
    if (viewMode === "scroll" && Date.now() < jump.current.until) scrollToPage(jump.current.page);
  }, [pageSizes, viewMode, scrollToPage]);

  const handleAreaScroll = useCallback(() => {
    if (viewMode !== "scroll" || scrollFrame.current) return;
    scrollFrame.current = requestAnimationFrame(() => {
      scrollFrame.current = 0;
      const area = containerRef.current;
      if (!area || Date.now() < jump.current.until) return;
      const areaTop = area.getBoundingClientRect().top;
      const tops: number[] = [];
      for (let page = 1; page <= numPages; page += 1) {
        const element = pageElements.current.get(page);
        if (!element) break;
        tops.push(element.getBoundingClientRect().top - areaTop + area.scrollTop);
      }
      const atBottom = area.scrollTop + area.clientHeight >= area.scrollHeight - 2;
      const page = atBottom && tops.length > 0 ? tops.length : pageAtProbe(tops, area.scrollTop + area.clientHeight * 0.4);
      if (page === pageNumberRef.current) return;
      pageFromScroll.current = true;
      pageNumberRef.current = page;
      setPageNumber(page);
      setPageInput(String(page));
    });
  }, [viewMode, numPages]);
  useEffect(() => () => cancelAnimationFrame(scrollFrame.current), []);
  // The reader taking over (wheel, touch, pointer) ends any jump still settling.
  const endJump = useCallback(() => { jump.current.until = 0; }, []);

  const recordPageSize = useCallback((page: number, size: PageSize) => {
    setPageSizes(current => {
      const known = current[page];
      return known && known.width === size.width && known.height === size.height ? current : { ...current, [page]: size };
    });
  }, []);

  // Continuous mode needs every page's size up front so the column's length, and
  // therefore every scroll position, is right before any page has been drawn.
  useEffect(() => {
    const pdf = pdfRef.current;
    if (viewMode !== "scroll" || !pdf || numPages === 0 || sizesLoadedFor.current === fileUrl) return;
    sizesLoadedFor.current = fileUrl;
    const token = ++sizeLoadToken.current;
    (async () => {
      for (let start = 1; start <= numPages; start += PAGE_SIZE_BATCH) {
        const count = Math.min(PAGE_SIZE_BATCH, numPages - start + 1);
        const batch = await Promise.all(Array.from({ length: count }, async (_, offset) => {
          const page = await pdf.getPage(start + offset);
          const viewport = page.getViewport({ scale: 1 });
          return [start + offset, { width: viewport.width, height: viewport.height }] as const;
        }));
        if (token !== sizeLoadToken.current) return;
        setPageSizes(current => ({ ...current, ...Object.fromEntries(batch) }));
      }
    })().catch(() => { if (token === sizeLoadToken.current) sizesLoadedFor.current = null; });
  }, [viewMode, numPages, fileUrl]);
  useEffect(() => () => { sizeLoadToken.current += 1; }, []);

  function chooseViewMode(mode: ViewMode) {
    setViewMode(mode);
    setSelectionMenu(null);
    try { window.localStorage.setItem(VIEW_MODE_STORAGE_KEY, mode); } catch { /* storage unavailable: the choice just isn't remembered */ }
  }
  // Back in single-page mode the page starts at its top, not wherever the column was scrolled.
  useEffect(() => { if (viewMode === "single") containerRef.current?.scrollTo({ top: 0 }); }, [viewMode]);

  function goToPage(next: number) {
    const clamped = Math.min(Math.max(1, next), numPages || next);
    setPageNumber(clamped);
    setPageInput(String(clamped));
    setSelectionMenu(null);
    scrollToPage(clamped); // a no-op outside scroll mode; needed when the page number itself doesn't change
  }
  function submitPageInput(event: React.FormEvent) {
    event.preventDefault();
    const parsed = Number.parseInt(pageInput, 10);
    if (Number.isFinite(parsed)) goToPage(parsed);
    else setPageInput(String(pageNumber));
  }
  // Zooming by hand leaves fit mode; the Fit button toggles it (off returns to the last manual zoom).
  function zoomIn() { setZoom(current => zoomBy(current, 1, fitScaleRef.current)); }
  function zoomOut() { setZoom(current => zoomBy(current, -1, fitScaleRef.current)); }
  function toggleFitToWidth() { setZoom(toggleFit); }
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

  // Where the active citation / search highlight sits on its page. Single-page
  // mode only has the current page to draw it on; continuous mode draws it on
  // whichever page it belongs to.
  const highlightBox = useMemo<HighlightBox | null>(() => {
    if (!activeHighlight || !activeHighlight.bbox) return null;
    if (viewMode === "single" && (!pageSize || activeHighlight.page !== pageNumber)) return null;
    const [x0, y0, x1, y1] = activeHighlight.bbox;
    return { left: x0 * scale, top: y0 * scale, width: Math.max(3, (x1 - x0) * scale), height: Math.max(3, (y1 - y0) * scale), search: activeHighlight.kind === "search" };
  }, [activeHighlight, viewMode, pageSize, pageNumber, scale]);

  useEffect(() => {
    if (!highlightBox) return;
    const frame = requestAnimationFrame(() => {
      highlightRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    return () => cancelAnimationFrame(frame);
  }, [highlightBox]);

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
          <button
            type="button"
            className={zoom.fit ? "is-active" : ""}
            onClick={toggleFitToWidth}
            aria-pressed={zoom.fit}
            title={zoom.fit ? "Fit to width is on — click to turn it off" : "Fit to width"}
            aria-label="Fit to width"
          ><FitWidthIcon /></button>
        </div>
        <div className="pdf-toolbar-group" role="group" aria-label="Page view">
          <button type="button" className={viewMode === "single" ? "is-active" : ""} onClick={() => chooseViewMode("single")} aria-pressed={viewMode === "single"} title="One page at a time" aria-label="One page at a time"><PageSingleIcon /></button>
          <button type="button" className={viewMode === "scroll" ? "is-active" : ""} onClick={() => chooseViewMode("scroll")} aria-pressed={viewMode === "scroll"} title="Continuous scroll" aria-label="Continuous scroll"><PageScrollIcon /></button>
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

      <div
        className={`pdf-canvas-area ${viewMode === "scroll" ? "is-continuous" : ""}`}
        ref={setAreaRef}
        onScroll={handleAreaScroll}
        onWheel={endJump}
        onTouchMove={endJump}
        onPointerDown={endJump}
      >
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
            onLoadSuccess={pdf => { pdfRef.current = pdf; setNumPages(pdf.numPages); setLoadError(""); }}
            onLoadError={() => setLoadError("This PDF could not be displayed.")}
            loading={<PdfSkeleton />}
            error={<div className="pdf-error">This PDF could not be displayed.</div>}
          >
            {viewMode === "single" ? (
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
                {highlightBox && (
                  <div
                    ref={highlightRef}
                    className={`pdf-highlight ${highlightBox.search ? "pdf-highlight-search" : ""}`}
                    style={{ left: highlightBox.left, top: highlightBox.top, width: highlightBox.width, height: highlightBox.height }}
                  />
                )}
              </div>
            ) : (
              <div className="pdf-scroll-column">
                {Array.from({ length: numPages }, (_, index) => {
                  const page = index + 1;
                  const size = sizeOf(page);
                  return (
                    <ScrollPage
                      key={page}
                      pageNumber={page}
                      scale={scale}
                      width={size.width * scale}
                      height={size.height * scale}
                      root={areaElement}
                      highlight={activeHighlight?.page === page ? highlightBox : null}
                      highlightRef={highlightRef}
                      registerElement={registerPageElement}
                      onSize={recordPageSize}
                    />
                  );
                })}
              </div>
            )}
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
