"use client";

import { assetUrl, Citation, VisualMeta } from "../../lib/api";
import { figureTitle, targetAsCitation, visualBanner } from "../../lib/visual";
import { ExternalLinkIcon, ImageIcon } from "./icons";

// Group 7 visual Q&A: what sits under a chart/figure answer -- an honest
// banner (unsupported, or "interpreted by <model>"), then the pages/figures
// concerned as real previews with one-click navigation into the PDF viewer.
// It only ever displays pixels the backend rendered; it never describes them.

export default function VisualPreview({ visual, onOpenCitation }: { visual: VisualMeta; onOpenCitation: (citation: Citation) => void }) {
  const banner = visualBanner(visual);
  return (
    <section className={`visual-card is-${banner.tone}`} aria-label="Visual elements referenced by this answer">
      <header className="visual-banner">
        <ImageIcon size={14} />
        <div>
          <strong>{banner.title}</strong>
          <p>{banner.detail}</p>
        </div>
      </header>
      {visual.targets.map(target => {
        const rasters = target.figures.filter(figure => figure.kind === "image");
        const captions = target.figures.filter(figure => figure.caption);
        return (
          <div className="visual-target" key={`${target.document_id}-${target.page_number}`}>
            <p className="visual-target-head"><span>{target.filename}</span><b>Page {target.page_number}</b></p>
            <div className="visual-thumbs">
              {rasters.map(figure => (
                <button
                  type="button"
                  key={figure.id}
                  className="visual-thumb"
                  onClick={() => onOpenCitation(targetAsCitation(target, figure.bbox))}
                  title={`${figureTitle(figure, target.page_number)} — open on page ${target.page_number}`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element -- preview is rendered by the API on demand; not a static asset next/image could optimize */}
                  <img src={assetUrl(figure.preview_url)} alt={`${figureTitle(figure, target.page_number)} preview`} loading="lazy" />
                  <span>{figureTitle(figure, target.page_number)}</span>
                </button>
              ))}
              <button
                type="button"
                className="visual-thumb is-page"
                onClick={() => onOpenCitation(targetAsCitation(target))}
                title={`Open page ${target.page_number}`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element -- same as above: API-rendered page image */}
                <img src={assetUrl(`${target.page_image_url}?dpi=60`)} alt={`Page ${target.page_number} preview`} loading="lazy" />
                <span>Page {target.page_number}</span>
              </button>
            </div>
            {captions.map(figure => <p className="visual-caption" key={figure.id}>{figure.caption}</p>)}
            <button type="button" className="open-in-pdf" onClick={() => onOpenCitation(targetAsCitation(target, rasters[0]?.bbox ?? null))}>
              <ExternalLinkIcon size={13} /> Open page {target.page_number} in PDF
            </button>
          </div>
        );
      })}
    </section>
  );
}
