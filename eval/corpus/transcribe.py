"""
Transcribe the corpus videos once and commit the transcripts.

    python -m eval.corpus.transcribe            # all videos missing a transcript
    python -m eval.corpus.transcribe --only mit_ai_lec1

Uses the backend's own yt-dlp download and faster-whisper transcription, so the
segments are exactly what the app would index. Committing them makes the
benchmark reproducible without rerunning Whisper (slow on CPU) and protects it
from yt-dlp breakage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from eval.bootstrap import setup_backend
from eval.corpus.download import load_manifest
from eval.paths import CORPUS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="transcribe just this video key")
    args = ap.parse_args()

    setup_backend()
    from config import settings
    from core.video_pipeline import transcribe_audio
    from utils.audio import download_audio_from_url

    for v in load_manifest()["videos"]:
        if args.only and v["key"] != args.only:
            continue
        out = CORPUS_DIR / v["transcript"]
        if out.exists():
            print(f"{v['key']}: transcript exists, skipping")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"{v['key']}: downloading audio ...", flush=True)
        wav, meta = download_audio_from_url(v["url"])
        try:
            t0 = time.time()
            print(f"{v['key']}: transcribing ({settings.WHISPER_MODEL}, CPU) ...", flush=True)
            segments = transcribe_audio(wav)
            data = {
                "key": v["key"],
                "url": v["url"],
                "title": v["title"],
                "duration_s": meta.get("duration") or v.get("duration_s"),
                "whisper_model": settings.WHISPER_MODEL,
                "translate_to_english": settings.TRANSLATE_TO_ENGLISH,
                "segments": segments,
            }
            out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{v['key']}: {len(segments)} segments in {time.time() - t0:.0f}s -> {out.name}")
        finally:
            if wav and os.path.exists(wav):
                os.remove(wav)
    return 0


if __name__ == "__main__":
    sys.exit(main())
