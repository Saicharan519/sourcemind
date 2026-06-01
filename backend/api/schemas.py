"""Request/response models."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# -- Ingestion --------------------------------------------------------------

class VideoIngestRequest(BaseModel):
    youtube_url: str = Field(..., description="Full YouTube URL")


class IngestResponse(BaseModel):
    source_id: UUID
    status: str
    message: str


# -- Sources ----------------------------------------------------------------

class SourceListItem(BaseModel):
    id: UUID
    source_type: str
    title: str
    status: str
    page_count: Optional[int] = None
    duration_s: Optional[int] = None
    eval_score: Optional[float] = None
    created_at: datetime


class EvalResultModel(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    overall_score: Optional[float] = None
    eval_questions: Optional[list[dict]] = None


class SourceDetail(BaseModel):
    id: UUID
    source_type: str
    title: str
    status: str
    page_count: Optional[int] = None
    duration_s: Optional[int] = None
    filename: Optional[str] = None
    youtube_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    eval_results: Optional[EvalResultModel] = None


class DeleteResponse(BaseModel):
    message: str


# -- Chat -------------------------------------------------------------------

class ChatHistoryItem(BaseModel):
    role: str
    content: str
