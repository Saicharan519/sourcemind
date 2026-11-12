"""Filesystem locations shared by every eval script. Pure stdlib, no backend imports."""
from __future__ import annotations

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_DIR = EVAL_DIR.parent
DEFAULT_BACKEND_DIR = REPO_DIR / "backend"

CORPUS_DIR = EVAL_DIR / "corpus"
MANIFEST = CORPUS_DIR / "manifest.yaml"
FILES_DIR = CORPUS_DIR / "files"              # downloaded PDFs (gitignored)
TRANSCRIPTS_DIR = CORPUS_DIR / "transcripts"  # committed video transcripts

DATASETS_DIR = EVAL_DIR / "datasets"
CONFIGS_DIR = EVAL_DIR / "configs"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"                # LLM response caches (gitignored)
WORK_DIR = EVAL_DIR / ".work"                 # scratch: uploads, cwd for backend (gitignored)
REPORT = EVAL_DIR / "REPORT.md"
