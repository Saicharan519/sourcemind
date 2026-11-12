"""
Validation for user-supplied video links (SSRF protection) and upload names.

Video links are fetched server-side by yt-dlp, so an attacker-controlled URL
could make the server request internal addresses (cloud metadata, localhost
services). The old check was a substring test, so
"http://169.254.169.254/?x=youtube.com" passed. Now the URL must be https and
its host must EXACTLY match an allow-listed public video host.
"""
from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit

ALLOWED_VIDEO_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "drive.google.com",
})


class UnsafeURLError(ValueError):
    pass


def validate_video_url(url: str) -> str:
    """Return the cleaned URL, or raise UnsafeURLError."""
    url = (url or "").strip()
    if not url or len(url) > 2048 or any(ch in url for ch in "\r\n\t "):
        raise UnsafeURLError("Provide a single YouTube or public Google Drive video link.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as e:
        raise UnsafeURLError("Malformed URL.") from e
    if parts.scheme != "https":
        raise UnsafeURLError("Only https links are accepted.")
    if parts.username or parts.password:
        raise UnsafeURLError("Links with embedded credentials are not accepted.")
    if port not in (None, 443):
        raise UnsafeURLError("Custom ports are not accepted.")
    host = (parts.hostname or "").lower().rstrip(".")
    if host not in ALLOWED_VIDEO_HOSTS:
        raise UnsafeURLError("Only YouTube and Google Drive video links are accepted.")
    return url


def safe_upload_name(original: str | None, allowed_ext: set[str]) -> str:
    """A server-chosen file name: random hex + the (validated, lower-case) extension.
    The user's file name never touches the filesystem path."""
    ext = os.path.splitext(original or "")[1].lower()
    if ext not in allowed_ext:
        raise ValueError(f"Unsupported file type: {ext or '(none)'}")
    return f"{uuid.uuid4().hex}{ext}"
