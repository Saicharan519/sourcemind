"""Audio acquisition helpers: yt-dlp downloads (YouTube/Drive) + ffmpeg extraction."""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import yt_dlp

from config import settings


def download_audio_from_url(url: str, out_dir: str | None = None) -> tuple[str, dict]:
    """
    Download best audio from a supported URL (YouTube or a public Google Drive
    video) as a 16 kHz WAV file via yt-dlp + ffmpeg.
    Returns (wav_path, metadata_dict) with title, duration (seconds), uploader.
    """
    out_dir = out_dir or settings.UPLOAD_DIR
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    out_template = os.path.join(out_dir, f"{job_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "extractor_retries": 3,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": ["-ar", "16000"],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    wav_path = os.path.join(out_dir, f"{job_id}.wav")
    if not os.path.exists(wav_path):
        for f in os.listdir(out_dir):
            if f.startswith(job_id):
                wav_path = os.path.join(out_dir, f)
                break

    meta = {
        "title": info.get("title", "Untitled video"),
        "duration": int(info.get("duration", 0) or 0),
        "uploader": info.get("uploader", ""),
    }
    return wav_path, meta


# Backwards-compatible alias.
download_youtube_audio = download_audio_from_url


def _ffprobe_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe (0.0 if it can't be read)."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", path,
            ],
            capture_output=True, text=True, timeout=120,
        )
        data = json.loads(out.stdout or "{}")
        return float(data.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        return 0.0


def extract_audio_from_file(video_path: str, out_dir: str | None = None) -> tuple[str, dict]:
    """
    Extract a 16 kHz mono WAV from an uploaded video file using ffmpeg.
    Returns (wav_path, metadata_dict) with title (from filename) and duration.
    """
    out_dir = out_dir or settings.UPLOAD_DIR
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    wav_path = os.path.join(out_dir, f"{job_id}.wav")
    duration = _ffprobe_duration(video_path)

    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",              # drop video
            "-ac", "1",         # mono
            "-ar", "16000",     # 16 kHz (what Whisper wants)
            "-f", "wav",
            wav_path,
        ],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0 or not os.path.exists(wav_path):
        tail = (proc.stderr or "")[-500:]
        raise RuntimeError(f"ffmpeg audio extraction failed: {tail}")

    meta = {
        "title": Path(video_path).stem,
        "duration": int(duration),
        "uploader": "",
    }
    return wav_path, meta
