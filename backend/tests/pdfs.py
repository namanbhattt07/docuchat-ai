from pathlib import Path

import fitz

# Small PDF builders for OCR / figure tests. A "scanned" page is a full-page
# raster image with NO text layer -- exactly what a scanner produces.

LOREM = (
    "Machine learning is a subfield of artificial intelligence concerned with algorithms that improve through "
    "experience. Supervised learning uses labelled examples, while unsupervised learning discovers structure."
)


def add_text_page(doc: "fitz.Document", text: str = LOREM, *, heading: str | None = "Introduction", size: float = 11) -> "fitz.Page":
    page = doc.new_page()
    y = 72
    if heading:
        page.insert_text((72, y), heading, fontsize=22)
        y += 36
    page.insert_textbox(fitz.Rect(72, y, 520, y + 200), text, fontsize=size)
    return page


def add_blank_page(doc: "fitz.Document") -> "fitz.Page":
    return doc.new_page()


def _image_png(text: str, *, width: int = 612, height: int = 792, dpi: int = 150, fill: tuple[float, float, float] | None = None) -> bytes:
    source = fitz.open()
    page = source.new_page(width=width, height=height)
    if fill:
        page.draw_rect(page.rect, color=None, fill=fill)
    if text:
        margin = min(72, min(width, height) // 8)
        page.insert_textbox(fitz.Rect(margin, margin, width - margin, height - margin), text, fontsize=16)
    return page.get_pixmap(dpi=dpi).tobytes("png")


def add_scanned_page(doc: "fitz.Document", text: str, *, rotation: int = 0) -> "fitz.Page":
    """A page whose only content is a picture of `text` (no text layer)."""
    page = doc.new_page()
    page.insert_image(page.rect, stream=_image_png(text))
    if rotation:
        page.set_rotation(rotation)
    return page


def add_logo_text_page(doc: "fitz.Document", text: str = LOREM) -> "fitz.Page":
    """Real native text plus a small decorative image -- must NOT be treated as scanned."""
    page = add_text_page(doc, text)
    page.insert_image(fitz.Rect(480, 20, 560, 60), stream=_image_png("", width=80, height=40, dpi=72, fill=(0.1, 0.2, 0.6)))
    return page


def add_figure_page(doc: "fitz.Page", caption: str = "Figure 1: Sensor throughput by protocol.") -> "fitz.Page":
    """Text page with a mid-sized embedded image (a chart-like figure) and a caption."""
    page = add_text_page(doc, "The chart below compares protocols.", heading="Results")
    page.insert_image(fitz.Rect(100, 200, 400, 420), stream=_image_png("", width=300, height=220, dpi=96, fill=(0.8, 0.3, 0.2)))
    page.insert_text((100, 440), caption, fontsize=10)
    return page


def _bar_chart_png() -> bytes:
    """A bar chart drawn purely as pixels: LoRa=5 (green), ZigBee=250 (orange), BLE=1000 (purple).
    The values and protocol names exist nowhere else in the PDF, so only something that can
    actually see the image can answer questions about them."""
    source = fitz.open()
    page = source.new_page(width=420, height=300)
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
    page.insert_text((90, 28), "Throughput by protocol (kbps)", fontsize=14, color=(0, 0, 0))
    page.draw_line((60, 250), (390, 250), color=(0, 0, 0), width=1.5)
    page.draw_line((60, 250), (60, 45), color=(0, 0, 0), width=1.5)
    for name, value, colour, x in (("LoRa", 5, (0.15, 0.65, 0.25), 90), ("ZigBee", 250, (0.95, 0.55, 0.1), 190), ("BLE", 1000, (0.5, 0.2, 0.75), 290)):
        height = max(4, value * 0.2)
        page.draw_rect(fitz.Rect(x, 250 - height, x + 60, 250), color=(0, 0, 0), fill=colour, width=0.8)
        page.insert_text((x + (2 if name == "ZigBee" else 8), 268), name, fontsize=11)
        page.insert_text((x + 8, 250 - height - 6), str(value), fontsize=11, color=(0, 0, 0))
    return page.get_pixmap(dpi=200).tobytes("png")


def add_bar_chart_page(doc: "fitz.Document") -> "fitz.Page":
    """Generic prose + the pixel-only bar chart + a caption. The text layer says nothing about the data."""
    page = doc.new_page()
    page.insert_text((72, 72), "Results", fontsize=22)
    page.insert_textbox(fitz.Rect(72, 100, 520, 150), "The chart below compares wireless protocols for the deployment.", fontsize=11)
    page.insert_image(fitz.Rect(90, 170, 500, 460), stream=_bar_chart_png())
    page.insert_text((90, 480), "Figure 1: Throughput comparison by protocol.", fontsize=10)
    return page


def build_pdf(path: Path, builders: list) -> Path:
    doc = fitz.open()
    for build in builders:
        build(doc)
    doc.save(path)
    doc.close()
    return path
