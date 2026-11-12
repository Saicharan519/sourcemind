"""
Download the evaluation corpus and verify every file against manifest.yaml.

    python -m eval.corpus.download          # download missing files, verify hashes
    python -m eval.corpus.download --lock   # record hashes for files that have none

A hash mismatch is a hard failure: the benchmark must not silently change.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request

import yaml

from eval.paths import FILES_DIR, MANIFEST

USER_AGENT = "Mozilla/5.0 (compatible; sourcemind-eval/1.0)"


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return yaml.safe_load(f)


def pdf_path(key: str):
    return FILES_DIR / f"{key}.pdf"


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 16):
            out.write(chunk)
    with open(tmp, "rb") as f:
        if f.read(5) != b"%PDF-":
            tmp.unlink()
            raise ValueError(f"{url} did not return a PDF")
    tmp.replace(dest)


def _record_hashes(hashes: dict[str, str]) -> None:
    """Write hashes into manifest.yaml in place, preserving comments and layout."""
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    current_key = None
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- key:"):
            current_key = stripped.split(":", 1)[1].strip()
        if stripped == "sha256: null" and current_key in hashes:
            line = line.replace("null", hashes[current_key])
        out.append(line)
    MANIFEST.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true", help="record hashes for unlocked files")
    args = ap.parse_args()

    FILES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    failures, to_lock = [], {}

    for doc in manifest["documents"]:
        dest = pdf_path(doc["key"])
        if not dest.exists():
            print(f"downloading {doc['key']} ...")
            _download(doc["url"], dest)
        digest = sha256_of(dest)
        expected = doc.get("sha256")
        if expected is None:
            to_lock[doc["key"]] = digest
            print(f"  {doc['key']}: {digest[:12]}... (not locked yet)")
        elif digest != expected:
            failures.append(doc["key"])
            print(f"  {doc['key']}: HASH MISMATCH (got {digest[:12]}..., want {expected[:12]}...)")
        else:
            print(f"  {doc['key']}: ok")

    if to_lock:
        if args.lock:
            _record_hashes(to_lock)
            print(f"locked {len(to_lock)} file(s) in {MANIFEST.name}")
        else:
            print(f"{len(to_lock)} file(s) unlocked; rerun with --lock to record their hashes")

    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
