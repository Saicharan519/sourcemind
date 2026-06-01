"""HuggingFace embedding wrapper (sentence-transformers all-MiniLM-L6-v2)."""
from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SentenceTransformer

from config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
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
