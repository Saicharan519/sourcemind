"""Centralized config loaded from env."""
from __future__ import annotations

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # LLM provider for the app: "groq" | "deepseek" | "cloudflare".
    # Exactly one provider is used; there is no automatic fallback, so answers never
    # silently come from a different model.
    LLM_PROVIDER: str = "groq"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL_MAIN: str = "openai/gpt-oss-120b"
    GROQ_MODEL_FAST: str = "openai/gpt-oss-20b"

    # DeepSeek (OpenAI-compatible API). Thinking mode is turned off: it adds latency
    # and billed reasoning tokens that a RAG answer does not need.
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL_MAIN: str = "deepseek-flash"
    DEEPSEEK_MODEL_FAST: str = "deepseek-flash"

    # Cloudflare Workers AI (OpenAI-compatible API, free daily Neuron allowance)
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""
    CLOUDFLARE_MODEL_MAIN: str = "@cf/openai/gpt-oss-120b"
    CLOUDFLARE_MODEL_FAST: str = "@cf/openai/gpt-oss-20b"

    # Mistral (used as the RAGAS evaluator LLM — higher free-tier limits than Groq)
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL_EVAL: str = "mistral-small-latest"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://sourcemind:sourcemind_secret@localhost:5432/sourcemind"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "sourcemind"

    # Whisper
    WHISPER_MODEL: str = "small"

    # Translate non-English sources to English at ingestion (chat is English-only).
    # Videos use Whisper's built-in translate task; PDFs are translated via Groq 8B.
    TRANSLATE_TO_ENGLISH: bool = True

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_DIM: int = 384

    # Hybrid search
    MMR_LAMBDA: float = 0.5
    MMR_FETCH_K: int = 20
    MMR_FINAL_K: int = 6
    BM25_TOP_K: int = 10
    DENSE_TOP_K: int = 10

    # Chunking
    PARENT_CHUNK_SIZE: int = 1500
    PARENT_CHUNK_OVERLAP: int = 200
    CHILD_CHUNK_SIZE: int = 300
    CHILD_CHUNK_OVERLAP: int = 30
    VIDEO_CHUNK_WORDS: int = 400

    # App
    MAX_PDF_SIZE_MB: int = 100
    MAX_VIDEO_SIZE_MB: int = 500
    MAX_VIDEO_DURATION_MIN: int = 120
    UPLOAD_DIR: str = "./uploads"
    BACKEND_CORS_ORIGINS: str = "http://localhost:5173"

    # RAGAS
    RAGAS_NUM_QUESTIONS: int = 10

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.BACKEND_CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
