"""Qdrant vector store wrapper."""
from __future__ import annotations

import uuid
from typing import Any, Optional
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from config import settings


class QdrantStore:
    _client: Optional[AsyncQdrantClient] = None

    @classmethod
    async def connect(cls) -> None:
        if cls._client is None:
            cls._client = AsyncQdrantClient(url=settings.QDRANT_URL)
            await cls._ensure_collection()

    @classmethod
    async def disconnect(cls) -> None:
        if cls._client is not None:
            await cls._client.close()
            cls._client = None

    @classmethod
    def client(cls) -> AsyncQdrantClient:
        if cls._client is None:
            raise RuntimeError("Qdrant not initialized. Call QdrantStore.connect() first.")
        return cls._client

    @classmethod
    async def _ensure_collection(cls) -> None:
        client = cls._client
        if client is None:
            return
        try:
            existing = await client.get_collections()
            names = {c.name for c in existing.collections}
            if settings.QDRANT_COLLECTION not in names:
                await client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=qm.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=qm.Distance.COSINE,
                    ),
                )
                # payload indexes for fast metadata filtering
                await client.create_payload_index(
                    collection_name=settings.QDRANT_COLLECTION,
                    field_name="source_id",
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
                await client.create_payload_index(
                    collection_name=settings.QDRANT_COLLECTION,
                    field_name="source_type",
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
        except Exception as e:
            print(f"[qdrant] ensure_collection failed: {e}")

    @classmethod
    async def upsert(
        cls,
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> list[str]:
        ids = [str(uuid.uuid4()) for _ in embeddings]
        points = [
            qm.PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(ids, embeddings, payloads)
        ]
        await cls.client().upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )
        return ids

    @classmethod
    async def search(
        cls,
        query_vector: list[float],
        source_id: UUID | None = None,
        top_k: int = 10,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        flt = None
        if source_id is not None:
            flt = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="source_id",
                        match=qm.MatchValue(value=str(source_id)),
                    )
                ]
            )
        results = await cls.client().search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [
            {
                "id": str(r.id),
                "score": float(r.score),
                "payload": r.payload or {},
                "vector": (list(r.vector) if with_vectors and r.vector is not None else None),
            }
            for r in results
        ]

    @classmethod
    async def retrieve(cls, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch points (payload + vector) by id. Used to backfill BM25-only hits
        that carry no Qdrant payload/vector after RRF fusion."""
        if not ids:
            return {}
        points = await cls.client().retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=ids,
            with_payload=True,
            with_vectors=True,
        )
        return {
            str(p.id): {
                "payload": p.payload or {},
                "vector": (list(p.vector) if p.vector is not None else None),
            }
            for p in points
        }

    @classmethod
    async def delete_by_source(cls, source_id: UUID) -> None:
        await cls.client().delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="source_id",
                            match=qm.MatchValue(value=str(source_id)),
                        )
                    ]
                )
            ),
        )
