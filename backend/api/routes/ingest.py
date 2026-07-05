"""Ingestion endpoints: PDF documents and YouTube videos."""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from api.schemas import IngestResponse, VideoIngestRequest
from config import settings
from core.document_pipeline import ingest_document
from core.evaluator import evaluate_source
from core.video_pipeline import ingest_video, ingest_video_file
from db.postgres import create_source, get_source

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# URL hosts we accept for link-based video ingestion (kept tight to avoid SSRF).
ALLOWED_VIDEO_HOSTS = ("youtube.com", "youtu.be", "drive.google.com")
# Video file extensions accepted for direct upload.
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


async def _ingest_doc_and_eval(source_id, pdf_path, filename):
    """Run ingestion, then trigger RAGAS eval if ingestion succeeded."""
    await ingest_document(source_id, pdf_path, filename)
    # Eval runs sequentially after ingestion. Failures are swallowed inside
    # evaluate_source, and score_with_ragas offloads the heavy work to a thread.
    if await _source_is_ready(source_id):
        await evaluate_source(source_id)


async def _ingest_video_and_eval(source_id, url):
    await ingest_video(source_id, url)
    if await _source_is_ready(source_id):
        await evaluate_source(source_id)


async def _ingest_video_file_and_eval(source_id, video_path, filename):
    await ingest_video_file(source_id, video_path, filename)
    if await _source_is_ready(source_id):
        await evaluate_source(source_id)


async def _source_is_ready(source_id) -> bool:
    """Only evaluate sources that finished ingesting (skip 'failed')."""
    src = await get_source(source_id)
    return bool(src and src.get("status") == "ready")


@router.post("/document", response_model=IngestResponse, status_code=202)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Persist upload to disk
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    pdf_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}_{file.filename}")

    # Read with size enforcement
    max_bytes = settings.MAX_PDF_SIZE_MB * 1024 * 1024
    total = 0
    with open(pdf_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                out.close()
                os.remove(pdf_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {settings.MAX_PDF_SIZE_MB} MB limit.",
                )
            out.write(chunk)

    # Create the source row (status=processing)
    source_id = await create_source(
        source_type="document",
        title=file.filename,
        filename=file.filename,
    )

    # Schedule background ingestion + eval
    background_tasks.add_task(_ingest_doc_and_eval, source_id, pdf_path, file.filename)

    return IngestResponse(
        source_id=source_id,
        status="processing",
        message="Document ingestion started.",
    )


@router.post("/video", response_model=IngestResponse, status_code=202)
async def ingest_youtube(
    payload: VideoIngestRequest,
    background_tasks: BackgroundTasks,
):
    url = payload.youtube_url.strip()
    if not url or not any(host in url for host in ALLOWED_VIDEO_HOSTS):
        raise HTTPException(
            status_code=400,
            detail="Provide a YouTube or public Google Drive video link.",
        )

    source_id = await create_source(
        source_type="video",
        title="Pending — fetching title",
        youtube_url=url,
    )

    background_tasks.add_task(_ingest_video_and_eval, source_id, url)

    return IngestResponse(
        source_id=source_id,
        status="processing",
        message="Video ingestion started.",
    )


@router.post("/video-file", response_model=IngestResponse, status_code=202)
async def ingest_video_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not file.filename or ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use mp4, mov, mkv, webm, avi, or m4v.",
        )

    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    video_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}_{file.filename}")

    max_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
    total = 0
    with open(video_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                out.close()
                os.remove(video_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {settings.MAX_VIDEO_SIZE_MB} MB limit.",
                )
            out.write(chunk)

    source_id = await create_source(
        source_type="video",
        title=file.filename,
        filename=file.filename,
    )

    background_tasks.add_task(
        _ingest_video_file_and_eval, source_id, video_path, file.filename
    )

    return IngestResponse(
        source_id=source_id,
        status="processing",
        message="Video ingestion started.",
    )
