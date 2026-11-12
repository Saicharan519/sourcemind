"""
Backend unit tests. No network, no databases, no LLM calls: every external
dependency is replaced with a fake via monkeypatch.

Run from the repo root with the backend venv:
    python -m pytest backend/tests -q
"""
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# config.py reads ".env" relative to the CWD and creates UPLOAD_DIR there.
# Run from an empty temp dir so no real .env (or API key) is ever loaded.
_TMP = tempfile.mkdtemp(prefix="sourcemind-tests-")
os.chdir(_TMP)
for k in ("GROQ_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "CLOUDFLARE_API_TOKEN"):
    os.environ[k] = "test-not-a-real-key"
os.environ.setdefault("LLM_PROVIDER", "groq")


# -- live-store fixture (real Postgres + Qdrant, isolated test DB/collection) ----

import asyncio  # noqa: E402

import pytest  # noqa: E402

TEST_DB = "sourcemind_test"
TEST_COLLECTION = "sourcemind_test"

DOC_A = [
    "Zorblax enzymes accelerate photosynthesis in deep-sea algae. "
    "The zorblax cycle was first measured at a rate of 42 molecules per second.",
    "Zorblax activity drops sharply below 4 degrees Celsius, which limits the algae to warm vents.",
]
DOC_B = [
    "Quantum lattice tariffs were introduced by the Veltrian trade council in 2031. "
    "The standard quantum lattice tariff is 17 percent.",
    "Exporters can apply for a lattice exemption if the shipment is under 200 kilograms.",
]


def _make_pdf(path, pages):
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=11)
    doc.save(str(path))
    doc.close()


@pytest.fixture(scope="session")
def live(tmp_path_factory):
    """Two tiny PDFs ingested through the REAL document pipeline into an isolated
    Postgres DB and Qdrant collection. Skips if the Docker databases are not running.
    Yields an object with .run(coro), .a, .b (source ids)."""
    import asyncpg

    from config import settings
    from core import document_pipeline
    from db.postgres import Database
    from db.qdrant import QdrantStore

    base = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1).rsplit("/", 1)[0]
    loop = asyncio.new_event_loop()

    async def setup():
        admin = await asyncpg.connect(base + "/postgres", timeout=3)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
            await admin.execute(f'CREATE DATABASE "{TEST_DB}"')
        finally:
            await admin.close()
        conn = await asyncpg.connect(f"{base}/{TEST_DB}")
        try:
            await conn.execute((BACKEND.parent / "scripts" / "init_db.sql").read_text(encoding="utf-8"))
        finally:
            await conn.close()

    try:
        loop.run_until_complete(setup())
    except Exception as e:  # DBs not running
        loop.close()
        pytest.skip(f"Postgres/Qdrant not available: {e}")

    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "DATABASE_URL", settings.DATABASE_URL.rsplit("/", 1)[0] + "/" + TEST_DB)
    mp.setattr(settings, "QDRANT_COLLECTION", TEST_COLLECTION)

    def no_llm():
        raise RuntimeError("no LLM in tests")  # title generation falls back to the filename

    mp.setattr(document_pipeline, "get_fast_llm", no_llm)

    async def ingest():
        Database._pool = None
        QdrantStore._client = None
        await Database.connect()
        await QdrantStore.connect()
        client = QdrantStore.client()
        names = {c.name for c in (await client.get_collections()).collections}
        if TEST_COLLECTION in names:
            await client.delete_collection(TEST_COLLECTION)
        await QdrantStore._ensure_collection()
        from db.postgres import create_source, get_source
        ids = {}
        for key, pages in (("a", DOC_A), ("b", DOC_B)):
            pdf = tmp_path_factory.mktemp("pdf") / f"{key}.pdf"
            _make_pdf(pdf, pages)
            sid = await create_source(source_type="document", title=f"doc_{key}.pdf", filename=f"doc_{key}.pdf")
            await document_pipeline.ingest_document(sid, str(pdf), f"doc_{key}.pdf")
            src = await get_source(sid)
            assert src["status"] == "ready", src
            ids[key] = sid
        return ids

    ids = loop.run_until_complete(ingest())

    class Live:
        a, b = ids["a"], ids["b"]

        @staticmethod
        def run(coro):
            return loop.run_until_complete(coro)

    yield Live

    async def teardown():
        await QdrantStore.client().delete_collection(TEST_COLLECTION)
        await QdrantStore.disconnect()
        await Database.disconnect()

    loop.run_until_complete(teardown())
    loop.close()
    mp.undo()
