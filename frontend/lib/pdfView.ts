// PDF viewer state rules, kept free of React and the DOM so they are unit-tested
// (tests/pdfview.test.mjs): how zoom and "Fit to width" interact, which reading
// mode is remembered, and which page a scroll position belongs to.

export type ViewMode = "single" | "scroll";

export const VIEW_MODE_STORAGE_KEY = "docuchat-pdf-view-mode";

export const MIN_SCALE = 0.4;
export const MAX_SCALE = 3;
export const ZOOM_STEP = 0.15;
/** What Fit to width returns to when it is switched off and nothing was zoomed by hand. */
export const ACTUAL_SIZE = 1;
/** Horizontal room kept around a page when fitting it to the viewer's width. */
export const FIT_GUTTER = 48;

export const clampScale = (value: number): number => Math.min(MAX_SCALE, Math.max(MIN_SCALE, value));

export const fitWidthScale = (containerWidth: number, pageWidth: number): number => clampScale((containerWidth - FIT_GUTTER) / pageWidth);

export const stepScale = (current: number, direction: 1 | -1): number => clampScale(Math.round((current + direction * ZOOM_STEP) * 100) / 100);

/** Anything but a stored "scroll" means the original one-page-at-a-time view. */
export const parseViewMode = (raw: string | null | undefined): ViewMode => (raw === "scroll" ? "scroll" : "single");

/** `fit` is a mode, not a zoom level: while it is on the scale follows the viewer's
 * width, and `manual` remembers the last zoom chosen by hand so switching fit off
 * has somewhere to go back to. */
export type ZoomState = { fit: boolean; manual: number };

export const initialZoom: ZoomState = { fit: true, manual: ACTUAL_SIZE };

/** The scale to draw at. `fitScale` is null until the page and viewer have been measured. */
export const effectiveScale = (state: ZoomState, fitScale: number | null): number => (state.fit && fitScale !== null ? fitScale : state.manual);

/** The Fit to width button: a real toggle. Off returns to the last manual zoom. */
export const toggleFit = (state: ZoomState): ZoomState => ({ ...state, fit: !state.fit });

/** Zooming by hand always leaves fit mode, continuing from the scale currently on screen. */
export const zoomBy = (state: ZoomState, direction: 1 | -1, fitScale: number | null): ZoomState => ({
  fit: false,
  manual: stepScale(effectiveScale(state, fitScale), direction),
});

/** The 1-based page a scroll position belongs to: the last page whose top is at or
 * above `probe`. `tops[i]` is the top of page i + 1, in the same coordinates as
 * `probe`, in ascending order. */
export function pageAtProbe(tops: number[], probe: number): number {
  if (tops.length === 0) return 1;
  let low = 0;
  let high = tops.length - 1;
  let found = 0;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (tops[middle] <= probe) { found = middle; low = middle + 1; }
    else high = middle - 1;
  }
  return found + 1;
}
