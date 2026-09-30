from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from the repository-level .env file."""

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DocuChat API"
    app_version: str = "0.2.0"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    database_url: str = "sqlite:///../data/docuchat.db"
    chroma_persist_directory: str = "../data/chroma"
    upload_directory: str = "../data/uploads"
    # Group 7: derived, regenerable files (page/figure preview renders) --
    # never the source of truth, so deleting this folder is always safe.
    processed_directory: str = "../data/processed"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_generation_model: str = "qwen3:8b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    # qwen3's hybrid "thinking" trace is the single biggest latency cost we
    # control (measured ~11x slower end to end) and the trace is discarded
    # anyway once the answer is parsed, so it is off by default.
    ollama_generation_think: bool = False
    ollama_keep_alive: str = "30m"
    # The context window every text-generation request asks Ollama for. Ollama's
    # own default is 4096 tokens, and a prompt longer than that is silently cut
    # down to ~half the window *from the front* -- which drops the instructions
    # and JSON schema and keeps only the tail of the passages. The largest
    # prompts this app builds (20-passage extraction ~5.7k tokens, 14-passage
    # templates, brainstorm/overview) overflow 4096, so it is pinned explicitly.
    # Keep it constant across requests: a different value makes Ollama reload
    # the model. It costs KV-cache memory (~0.15 MB/token for qwen3:8b).
    ollama_num_ctx: int = 8192
    ollama_embed_timeout_seconds: float = 90
    ollama_generate_timeout_seconds: float = 180
    ollama_embed_concurrency: int = 4
    # Group 7: optional local *vision* model for visual Q&A. Empty (the
    # default) means "none" -- the text model qwen3:8b cannot read images and
    # is never used as a stand-in. Set to an installed vision-capable Ollama
    # model (e.g. one that reports the "vision" capability) to enable it;
    # DocuChat never downloads a model on its own.
    ollama_vision_model: str = ""

    max_upload_size_mb: int = 50
    chunk_size: int = 900
    chunk_overlap: int = 150

    retrieval_top_k: int = 8
    # Widens the candidate search set only (pure vector/lexical scoring, no
    # extra LLM calls) -- 24 was thin once documents run into the hundreds
    # of pages and thousands of chunks. retrieval_top_k (the final context
    # fed to the model) stays at 8 so generation latency doesn't grow.
    retrieval_candidate_pool: int = 40
    retrieval_overview_max_sources: int = 10

    # Hybrid retrieval (Group 3): reciprocal-rank-fusion weights for the two
    # candidate lists. Vector carries a bit more weight by default -- it is
    # the signal that actually understands paraphrase/synonymy -- while
    # keyword/FTS5 is what rescues exact terms, identifiers, and proper
    # nouns a paraphrase-tuned embedding can under-rank.
    retrieval_vector_weight: float = 1.0
    retrieval_keyword_weight: float = 0.8
    retrieval_rrf_k: int = 60
    retrieval_section_max_sources: int = 8
    retrieval_page_max_sources: int = 8
    retrieval_multi_document_max_sources: int = 12
    retrieval_selection_max_sources: int = 6
    # Group 4: LOCATION is a lookup, not a synthesis task, so it only needs a
    # handful of distinct page/section hits. EXTRACTION needs much wider
    # coverage since the entities being extracted (e.g. every protocol) can
    # be scattered across the whole document. CALCULATION only needs enough
    # passages to plausibly contain the 1-2 numeric values being compared.
    retrieval_location_max_sources: int = 6
    retrieval_extraction_max_sources: int = 20
    retrieval_calculation_max_sources: int = 6
    # Chunks are considered near-duplicates (and collapsed to the
    # higher-scored one) when they overlap this fraction of their combined
    # character span on the same page.
    retrieval_dedup_overlap_ratio: float = 0.6

    suggested_questions_count: int = 5

    # Group 7 OCR (see services/ocr.py, services/pdf_provider.py). A page is
    # OCR'd only when its native text layer is thin (fewer than
    # ocr_min_native_chars letters/digits) AND raster images cover at least
    # ocr_min_image_coverage of it -- an image alone never triggers OCR, and
    # a page with real native text is never OCR'd.
    ocr_enabled: bool = True
    ocr_min_native_chars: int = 30
    ocr_min_image_coverage: float = 0.5
    # Render resolution for OCR input, capped by the longest side in pixels
    # so a poster-sized page can't allocate a huge bitmap.
    ocr_render_dpi: int = 200
    ocr_max_image_px: int = 2400
    # OCR lines the engine is less sure of than this are dropped as noise.
    ocr_min_confidence: float = 0.5
    # Upper bound on OCR'd pages per document -- OCR is slow, so a 1,000 page
    # scan is capped instead of silently tying up the machine for hours.
    ocr_max_pages: int = 150

    # Group 7 figure handling. Raster images smaller than this on either side
    # (px) or covering less than image_min_area_ratio of the page are treated
    # as decorative (icons, rules, bullets) and not recorded as figures.
    image_min_dimension_px: int = 64
    image_min_area_ratio: float = 0.01
    # An image covering at least this share of the page is a full-page scan
    # or background, not a figure -- unless the page turns out to hold no
    # readable text at all (a full-page photo/artwork), see pdf_provider.
    image_max_area_ratio: float = 0.9
    page_render_default_dpi: int = 110
    page_render_max_dpi: int = 200

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
