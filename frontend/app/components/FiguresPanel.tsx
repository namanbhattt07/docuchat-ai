"use client";

import { useEffect, useMemo, useState } from "react";
import { assetUrl, BBox, DocumentImageItem, DocumentPagesResponse, getDocumentImages, getDocumentPages } from "../../lib/api";
import { pageSummaryText } from "../../lib/ocr";
import { figureTitle } from "../../lib/visual";
import { CloseIcon, ImageIcon } from "./icons";

// Group 7 image/figure browser: the figures and images detected in the open
// document, with a preview of each and of its full page. Metadata and
// rendering only -- nothing here interprets an image.

type Props = {
  documentId: string;
  currentPage: number;
  onNavigate: (page: number, bbox?: BBox | null) => void;
  onClose: () => void;
};

export default function FiguresPanel({ documentId, currentPage, onNavigate, onClose }: Props) {
  const [images, setImages] = useState<DocumentImageItem[] | null>(null);
  const [pages, setPages] = useState<DocumentPagesResponse | null>(null);
  const [error, setError] = useState("");
  const [showPageImage, setShowPageImage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setImages(null); setPages(null); setError(""); setShowPageImage(null);
    Promise.all([getDocumentImages(documentId), getDocumentPages(documentId)])
      .then(([imageResult, pageResult]) => { if (!cancelled) { setImages(imageResult.images); setPages(pageResult); } })
      .catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : "Could not load the figures."); });
    return () => { cancelled = true; };
  }, [documentId]);

  const summary = useMemo(() => {
    if (!pages) return null;
    const counts = { text: 0, ocr: 0, empty: 0, failed: 0, unknown: 0 };
    for (const page of pages.pages) counts[page.status] += 1;
    return pageSummaryText(counts, pages.page_count);
  }, [pages]);

  return (
    <div className="toc-panel figures-panel">
      <header className="toc-header">
        <span className="toc-heading"><ImageIcon size={15} /> Figures &amp; pages</span>
        <button type="button" className="toc-close" onClick={onClose} aria-label="Close figures panel"><CloseIcon size={13} /></button>
      </header>
      <div className="toc-body">
        {summary && <p className="figures-summary">{summary}</p>}
        {error && <div className="toc-state toc-error">{error}</div>}
        {!error && images === null && <div className="toc-skeleton"><div /><div /><div /></div>}
        {!error && images !== null && images.length === 0 && (
          <div className="toc-state toc-empty">No figures or images were detected in this document.</div>
        )}
        {!error && images !== null && images.length > 0 && (
          <ul className="figures-list">
            {images.map(image => {
              const title = figureTitle(image, image.page_number);
              return (
                <li key={image.id} className={`figure-card ${image.page_number === currentPage ? "is-current" : ""}`}>
                  <div className="figure-card-head">
                    <b>{title}</b>
                    <span>Page {image.page_number}</span>
                  </div>
                  {image.kind === "image" && (
                    // eslint-disable-next-line @next/next/no-img-element -- API-rendered crop; not a static asset next/image could optimize
                    <img className="figure-preview" src={assetUrl(image.preview_url)} alt={`${title} preview`} loading="lazy" />
                  )}
                  {image.caption && <p className="figure-caption">{image.caption}</p>}
                  {image.kind === "image" && image.width && image.height && <p className="figure-dims">{image.width} × {image.height} px source image</p>}
                  {showPageImage === image.id && (
                    // eslint-disable-next-line @next/next/no-img-element -- API-rendered page image
                    <img className="figure-page-image" src={assetUrl(`${image.page_image_url}?dpi=80`)} alt={`Page ${image.page_number}`} loading="lazy" />
                  )}
                  <div className="figure-actions">
                    <button type="button" onClick={() => onNavigate(image.page_number, image.kind === "image" ? image.bbox : null)}>Go to page {image.page_number}</button>
                    <button type="button" onClick={() => setShowPageImage(current => (current === image.id ? null : image.id))}>
                      {showPageImage === image.id ? "Hide page image" : "Page image"}
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
