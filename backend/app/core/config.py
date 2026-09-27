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

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_generation_model: str = "qwen3:8b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    # qwen3's hybrid "thinking" trace is the single biggest latency cost we
    # control (measured ~11x slower end to end) and the trace is discarded
    # anyway once the answer is parsed, so it is off by default.
    ollama_generation_think: bool = False
    ollama_keep_alive: str = "30m"
    ollama_embed_timeout_seconds: float = 90
    ollama_generate_timeout_seconds: float = 180
    ollama_embed_concurrency: int = 4

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

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
