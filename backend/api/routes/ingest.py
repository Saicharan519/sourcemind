"""Ingestion endpoints: PDF documents and YouTube videos."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from api.schemas import IngestResponse, VideoIngestRequest
from config import settings
from core.document_pipeline import ingest_document
from core.evaluator import evaluate_source
from core.video_pipeline import ingest_video, ingest_video_file
from db.postgres import create_source, get_source
from utils.url_safety import UnsafeURLError, safe_upload_name, validate_video_url

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# Video file extensions accepted for direct upload.
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


async def _save_upload(file: UploadFile, allowed_ext: set[str], max_mb: int,
                       magic: bytes | None = None) -> str:
    """Stream an upload to UPLOAD_DIR under a server-chosen name, enforcing the size
    limit and (optionally) the file's magic bytes. Returns the saved path."""
    try:
        name = safe_upload_name(file.filename, allowed_ext)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    path = os.path.join(settings.UPLOAD_DIR, name)
    max_bytes = max_mb * 1024 * 1024
    total, first = 0, True
    try:
        with open(path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                if first and magic and not chunk.startswith(magic):
                    raise HTTPException(status_code=400, detail="File content does not match its type.")
                first = False
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File exceeds {max_mb} MB limit.")
                out.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty file.")
    except HTTPException:
        if os.path.exists(path):
            os.remove(path)
        raise
    return path


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


def _display_name(filename: str | None) -> str:
    """The user's file name, for display only (never used in a filesystem path)."""
    return os.path.basename((filename or "").replace("\\", "/"))[:200] or "upload"


@router.post("/document", response_model=IngestResponse, status_code=202)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    pdf_path = await _save_upload(file, {".pdf"}, settings.MAX_PDF_SIZE_MB, magic=b"%PDF-")
    name = _display_name(file.filename)
    source_id = await create_source(source_type="document", title=name, filename=name)
    background_tasks.add_task(_ingest_doc_and_eval, source_id, pdf_path, name)
    return IngestResponse(source_id=source_id, status="processing",
                          message="Document ingestion started.")


@router.post("/video", response_model=IngestResponse, status_code=202)
async def ingest_youtube(
    payload: VideoIngestRequest,
    background_tasks: BackgroundTasks,
):
    try:
        url = validate_video_url(payload.youtube_url)
    except UnsafeURLError as e:
        raise HTTPException(status_code=400, detail=str(e))
    source_id = await create_source(source_type="video", title="Pending - fetching title",
                                    youtube_url=url)
    background_tasks.add_task(_ingest_video_and_eval, source_id, url)
    return IngestResponse(source_id=source_id, status="processing",
                          message="Video ingestion started.")


@router.post("/video-file", response_model=IngestResponse, status_code=202)
async def ingest_video_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    video_path = await _save_upload(file, ALLOWED_VIDEO_EXT, settings.MAX_VIDEO_SIZE_MB)
    name = _display_name(file.filename)
    source_id = await create_source(source_type="video", title=name, filename=name)
    background_tasks.add_task(_ingest_video_file_and_eval, source_id, video_path, name)
    return IngestResponse(source_id=source_id, status="processing",
                          message="Video ingestion started.")
