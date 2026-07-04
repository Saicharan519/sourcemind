"""
RAGAS evaluator.

For a newly ingested source:
  1. Sample chunks, generate 10 QA pairs via Groq 8B
  2. For each Q, run the full RAG pipeline to get the generated answer + contexts
  3. Score with RAGAS (faithfulness, answer_relevancy, context_recall, context_precision)
  4. Store results in postgres eval_results
"""
from __future__ import annotations

import json
import math
from typing import Any
from uuid import UUID

from config import settings
from core.rag_engine import (
    generate_answer_sync,
    get_fast_llm,
    hydrate_contexts,
    retrieve,
)
from db.postgres import Database, insert_eval_result
from ragas.run_config import RunConfig


# -- QA generation ---------------------------------------------------------

QA_GEN_PROMPT = """\
You are creating an evaluation dataset for a RAG system. Given the following
content from a source, generate {n} diverse question/answer pairs that a real
user might ask. Each pair must be fully answerable from the content alone.

Return STRICT JSON: a JSON array of objects, each with "question" and
"ground_truth" string fields. No other text, no markdown.

Content:
{content}
"""


async def _sample_source_content(source_id: UUID, n_chunks: int = 6) -> str:
    """Pull a diverse text sample from the source's parent chunks (docs) or BM25
    corpus (videos and fallback)."""
    rows = await Database.fetch(
        """
        SELECT content
        FROM parent_chunks
        WHERE source_id = $1
        ORDER BY chunk_index
        LIMIT $2
        """,
        source_id, n_chunks,
    )
    if rows:
        return "\n\n---\n\n".join(r["content"] for r in rows)

    rows = await Database.fetch(
        """
        SELECT content
        FROM bm25_index
        WHERE source_id = $1
        ORDER BY chunk_index
        LIMIT $2
        """,
        source_id, n_chunks,
    )
    return "\n\n---\n\n".join(r["content"] for r in rows)


async def generate_qa_pairs(source_id: UUID, n: int) -> list[dict[str, str]]:
    content = await _sample_source_content(source_id)
    if not content.strip():
        return []

    # truncate to keep within token budget
    content = content[:6000]

    llm = get_fast_llm()
    resp = await llm.ainvoke(QA_GEN_PROMPT.format(n=n, content=content))
    text = resp.content if hasattr(resp, "content") else str(resp)

    # extract JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        print("[ragas] QA generation returned no JSON array")
        return []

    try:
        pairs = json.loads(text[start : end + 1])
    except Exception as e:
        print(f"[ragas] JSON parse failed: {e}")
        return []

    cleaned = []
    for p in pairs:
        q = (p.get("question") or "").strip()
        gt = (p.get("ground_truth") or p.get("answer") or "").strip()
        if q and gt:
            cleaned.append({"question": q, "ground_truth": gt})
    return cleaned[:n]


# -- RAG run over QA set ---------------------------------------------------

async def run_rag_on_qa(source_id: UUID, qa_pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """For each question, run retrieve+answer. Collect contexts and generated answers."""
    records = []
    for pair in qa_pairs:
        q = pair["question"]
        try:
            chunks = await retrieve(q, source_id=source_id)
            chunks = await hydrate_contexts(chunks)
            contexts = [c.get("content", "") for c in chunks if c.get("content")]
            answer = await generate_answer_sync(q, chunks)
            records.append({
                "question": q,
                "ground_truth": pair["ground_truth"],
                "answer": answer,
                "contexts": contexts[:6],
            })
        except Exception as e:
            print(f"[ragas] RAG run failed for question '{q[:50]}...': {e}")
            continue
    return records


# -- RAGAS scoring ---------------------------------------------------------

def _build_ragas_llm():
    """
    Wrap the RAGAS evaluator LLM. Prefer Mistral (higher free-tier limits, which
    avoids the 429 → timeout → NaN cascade that Groq's low TPM caused). Falls
    back to Groq 8B if no Mistral key is configured.
    """
    from ragas.llms import LangchainLLMWrapper

    if settings.MISTRAL_API_KEY:
        from langchain_mistralai import ChatMistralAI
        chat = ChatMistralAI(
            model=settings.MISTRAL_MODEL_EVAL,
            api_key=settings.MISTRAL_API_KEY,
            temperature=0.0,
        )
    else:
        from langchain_groq import ChatGroq
        print("[ragas] MISTRAL_API_KEY not set — falling back to Groq 8B evaluator")
        chat = ChatGroq(
            model=settings.GROQ_MODEL_FAST,
            api_key=settings.GROQ_API_KEY,
            temperature=0.0,
        )
    return LangchainLLMWrapper(chat)


def _build_ragas_embeddings():
    """Wrap sentence-transformers HF embeddings for RAGAS."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    emb = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
    )
    return LangchainEmbeddingsWrapper(emb)


async def score_with_ragas(records: list[dict[str, Any]]) -> dict[str, float]:
    """Returns aggregate scores across the dataset."""
    if not records:
        return {}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        print(f"[ragas] import failed: {e}")
        return {}

    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in records
    ])

    try:
        ragas_llm = _build_ragas_llm()
        ragas_emb = _build_ragas_embeddings()

        import ragas as _ragas
        _ragas.utils.num_workers = 1

        # ragas.evaluate() is synchronous and CPU/network-heavy — run it off the
        # event loop so eval never freezes the API server.
        def _run_eval():
            result = evaluate(
                ds,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                llm=ragas_llm,
                embeddings=ragas_emb,
                raise_exceptions=False,
                run_config=RunConfig(timeout=120, max_retries=3, max_wait=60),
            )
            return result.to_pandas()

        import asyncio
        df = await asyncio.to_thread(_run_eval)
        return {
            "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else None,
            "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else None,
            "context_recall": float(df["context_recall"].mean()) if "context_recall" in df else None,
            "context_precision": float(df["context_precision"].mean()) if "context_precision" in df else None,
        }
    except Exception as e:
        print(f"[ragas] evaluation failed: {e}")
        return {}


# -- Public entrypoint -----------------------------------------------------

async def evaluate_source(source_id: UUID) -> dict[str, Any]:
    """
    Full eval pipeline: QA gen → RAG runs → RAGAS scoring → persist.
    Safe to call as a BackgroundTask. Returns the result dict.
    """
    try:
        pairs = await generate_qa_pairs(source_id, n=settings.RAGAS_NUM_QUESTIONS)
        if not pairs:
            print(f"[ragas] no QA pairs generated for {source_id}, skipping eval")
            return {}

        records = await run_rag_on_qa(source_id, pairs)
        if not records:
            return {}

        scores = await score_with_ragas(records)

        valid_scores = [v for v in scores.values() if isinstance(v, (int, float)) and v is not None]
        overall = sum(valid_scores) / len(valid_scores) if valid_scores else None
        if overall is not None and math.isnan(overall):
            overall = None
        eval_questions = [
            {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "generated_answer": r["answer"],
            }
            for r in records
        ]

        result = {
            **scores,
            "overall_score": overall,
            "eval_questions": eval_questions,
        }
        await insert_eval_result(source_id, result)
        print(f"[ragas] eval done for {source_id}, overall={overall}")
        return result
    except Exception as e:
        print(f"[ragas] evaluate_source FAILED for {source_id}: {e}")
        return {}
