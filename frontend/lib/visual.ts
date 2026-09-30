import type { BBox, Citation, VisualMeta, VisualTarget } from "./api";

// Group 7 visual Q&A presentation helpers. The one rule they encode: the UI
// only ever says a model interpreted an image when the backend reports
// `supported: true` with the model's name -- otherwise it shows the honest
// "not available" state.

export type VisualBanner = { tone: "unsupported" | "interpreted"; title: string; detail: string };

export function visualBanner(visual: VisualMeta): VisualBanner {
  if (visual.supported && visual.model) {
    return {
      tone: "interpreted",
      title: `Interpreted by local vision model ${visual.model}`,
      detail: "Generated from the rendered page image; it may be inaccurate. Check it against the page.",
    };
  }
  return {
    tone: "unsupported",
    title: "Visual questions aren't supported by the current model setup",
    detail: visual.reason,
  };
}

/** A Citation-shaped handle so a preview can reuse the app's existing
 * "open this page (and highlight this box) in the PDF viewer" navigation. */
export function targetAsCitation(target: VisualTarget, bbox: BBox | null = null): Citation {
  return {
    index: 0,
    document_id: target.document_id,
    filename: target.filename,
    page_number: target.page_number,
    excerpt: "",
    bbox,
    bbox_source: bbox ? "exact" : "unavailable",
  };
}

export const figureTitle = (figure: { label: string | null; kind: string }, pageNumber: number): string =>
  figure.label ?? (figure.kind === "image" ? `Image on page ${pageNumber}` : `Figure on page ${pageNumber}`);
