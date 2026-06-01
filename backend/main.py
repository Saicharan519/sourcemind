"""SourceMind backend entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, evaluate, global_chat, ingest, sources
from config import settings
from db.postgres import Database
from db.qdrant import QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[lifespan] connecting to Postgres...")
    await Database.connect()
    print("[lifespan] connecting to Qdrant...")
    await QdrantStore.connect()
    print("[lifespan] ready.")
    yield
    # Shutdown
    print("[lifespan] shutting down...")
    await Database.disconnect()
    await QdrantStore.disconnect()


app = FastAPI(
    title="SourceMind",
    description="Multi-modal RAG over documents and YouTube videos.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
app.include_router(ingest.router)
app.include_router(sources.router)
app.include_router(chat.router)
app.include_router(global_chat.router)
app.include_router(evaluate.router)


@app.get("/")
async def root():
    return {
        "service": "SourceMind",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
