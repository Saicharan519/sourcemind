import sys

# Windows consoles/redirects default to cp1252, which crashes on characters that
# PDFs and LLM output use all the time (non-breaking hyphens, ellipses, ...).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
