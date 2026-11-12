"""
Make the SourceMind backend importable from eval scripts, pointed at an
*isolated* eval database and Qdrant collection.

Call `setup_backend()` BEFORE importing anything from the backend (config.py
reads the environment at import time).

Why the care:
- The backend's config loads `.env` from the current working directory. The repo
  root has a stale `.env`, so we chdir into an empty scratch dir and instead copy
  the keys from `backend/.env` into os.environ ourselves.
- The eval must never touch the app's real data, so DATABASE_URL and
  QDRANT_COLLECTION are overridden to `sourcemind_eval`.
- `backend_dir` can point at another checkout (e.g. a `v1-as-shipped` worktree)
  so old code can be measured with the same harness.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from eval.paths import DEFAULT_BACKEND_DIR, WORK_DIR

EVAL_DB = "sourcemind_eval"
EVAL_COLLECTION = "sourcemind_eval"


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split(" #", 1)[0].strip().strip('"').strip("'")
        values[key.strip()] = val
    return values


def setup_backend(backend_dir: str | Path | None = None, overrides: dict | None = None) -> Path:
    backend = Path(backend_dir or DEFAULT_BACKEND_DIR).resolve()
    if not (backend / "config.py").exists():
        raise FileNotFoundError(f"no backend at {backend}")

    # Secrets and model settings always come from the main checkout's backend/.env,
    # so a worktree of old code (which has no .env) still has API keys.
    for k, v in _read_dotenv(DEFAULT_BACKEND_DIR / ".env").items():
        os.environ.setdefault(k, v)

    base_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://sourcemind:sourcemind_secret@localhost:5432/sourcemind",
    )
    os.environ["DATABASE_URL"] = base_url.rsplit("/", 1)[0] + f"/{EVAL_DB}"
    os.environ["QDRANT_COLLECTION"] = EVAL_COLLECTION
    os.environ["UPLOAD_DIR"] = str(WORK_DIR / "uploads")
    for k, v in (overrides or {}).items():
        os.environ[str(k)] = str(v)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK_DIR)  # no .env here, so config.py only sees os.environ
    sys.path.insert(0, str(backend))
    return backend


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty(path: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--", "."], text=True
        )
        return bool(out.strip())
    except Exception:
        return True
