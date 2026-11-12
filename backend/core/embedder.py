"""HuggingFace embedding wrapper (sentence-transformers all-MiniLM-L6-v2)."""
from __future__ import annotations

import threading
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import settings


_LOAD_LOCK = threading.Lock()


def get_embedder() -> SentenceTransformer:
    """Load the embedding model once. The first query embeds several query variants
    in parallel threads; without the lock each one loaded its own copy of the model."""
    with _LOAD_LOCK:
        return _load_embedder()


@lru_cache(maxsize=1)
def _load_embedder() -> SentenceTransformer:
    print(f"[embedder] loading model: {settings.EMBEDDING_MODEL} on {settings.EMBEDDING_DEVICE}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
    print("[embedder] model loaded.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
