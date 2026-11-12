"""SSRF protection, upload handling and API error paths (no databases needed)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import ingest
from utils.url_safety import UnsafeURLError, safe_upload_name, validate_video_url


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc123",
    "https://youtu.be/abc123",
    "https://drive.google.com/file/d/xyz/view",
    "https://m.youtube.com/watch?v=abc",
])
def test_accepts_allowed_video_hosts(url):
    assert validate_video_url(url) == url


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/?x=youtube.com",   # old substring check passed this
    "https://169.254.169.254/?v=youtube.com",
    "https://evil.com/youtube.com/watch",
    "https://youtube.com.evil.com/watch?v=1",
    "https://notyoutube.com/watch?v=1",
    "http://www.youtube.com/watch?v=1",                         # not https
    "https://user:pw@www.youtube.com/watch?v=1",                # credentials
    "https://www.youtube.com:8080/watch?v=1",                   # custom port
    "https://localhost/watch",
    "file:///etc/passwd",
    "",
    "https://www.youtube.com/watch?v=1 https://evil.com",
])
def test_rejects_unsafe_video_urls(url):
    with pytest.raises(UnsafeURLError):
        validate_video_url(url)


def test_upload_name_is_server_chosen():
    name = safe_upload_name("../../etc/evil.PDF", {".pdf"})
    assert name.endswith(".pdf") and "/" not in name and ".." not in name and len(name) == 36
    with pytest.raises(ValueError):
        safe_upload_name("payload.exe", {".pdf"})


@pytest.fixture
def client(monkeypatch, tmp_path):
    created, scheduled = [], []

    async def fake_create_source(**kw):
        created.append(kw)
        return "00000000-0000-0000-0000-000000000001"

    async def noop(*a, **kw):
        scheduled.append(a)

    monkeypatch.setattr(ingest, "create_source", fake_create_source)
    monkeypatch.setattr(ingest, "_ingest_doc_and_eval", noop)
    monkeypatch.setattr(ingest, "_ingest_video_and_eval", noop)
    monkeypatch.setattr(ingest.settings, "UPLOAD_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(ingest.router)
    c = TestClient(app)
    c.created, c.scheduled, c.dir = created, scheduled, tmp_path
    return c


def test_pdf_upload_saved_under_safe_name(client):
    r = client.post("/api/ingest/document",
                    files={"file": ("../../secret report.pdf", b"%PDF-1.4 tiny", "application/pdf")})
    assert r.status_code == 202, r.text
    saved = list(client.dir.iterdir())
    assert len(saved) == 1 and saved[0].suffix == ".pdf" and "secret" not in saved[0].name
    assert client.created[0]["title"] == "secret report.pdf"   # display name only


def test_fake_pdf_rejected_and_not_kept(client):
    r = client.post("/api/ingest/document",
                    files={"file": ("doc.pdf", b"MZ\x90\x00 not a pdf", "application/pdf")})
    assert r.status_code == 400
    assert list(client.dir.iterdir()) == [] and client.created == []


def test_non_pdf_extension_rejected(client):
    r = client.post("/api/ingest/document", files={"file": ("doc.txt", b"%PDF-", "text/plain")})
    assert r.status_code == 400


def test_video_ssrf_rejected_by_api(client):
    r = client.post("/api/ingest/video",
                    json={"youtube_url": "http://169.254.169.254/latest/meta-data/?x=youtube.com"})
    assert r.status_code == 400 and client.created == []
    r = client.post("/api/ingest/video", json={"youtube_url": "https://youtu.be/abc123"})
    assert r.status_code == 202
