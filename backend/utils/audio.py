"""yt-dlp + pydub audio acquisition helpers."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import yt_dlp

from config import settings


def download_youtube_audio(url: str, out_dir: str | None = None) -> tuple[str, dict]:
    """
    Download best audio from a YouTube URL as a WAV file.
    Returns (wav_path, metadata_dict).
    metadata_dict includes: title, duration (seconds), uploader.
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

    # After post-processing, the file is .wav
    wav_path = os.path.join(out_dir, f"{job_id}.wav")
    if not os.path.exists(wav_path):
        # fallback: scan for any file with the job_id prefix
        for f in os.listdir(out_dir):
            if f.startswith(job_id):
                wav_path = os.path.join(out_dir, f)
                break

    meta = {
        "title": info.get("title", "Untitled video"),
        "duration": int(info.get("duration", 0)),
        "uploader": info.get("uploader", ""),
    }
    return wav_path, meta
