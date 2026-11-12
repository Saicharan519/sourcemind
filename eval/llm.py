"""
LLM helpers for the eval tooling (question generation, cross-checking, judging).

- Keys and model names come from backend/.env (no backend import needed).
- Every call is cached on disk (sqlite) keyed by model + prompt + temperature, so
  re-running a script is free and deterministic.
- 429 / transient errors are retried with exponential backoff, and calls are
  throttled to stay under free-tier rate limits.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time

from eval.bootstrap import _read_dotenv
from eval.paths import CACHE_DIR, DEFAULT_BACKEND_DIR

_ENV = _read_dotenv(DEFAULT_BACKEND_DIR / ".env")


def env(key: str, default: str = "") -> str:
    import os
    return os.environ.get(key) or _ENV.get(key) or default


GROQ_FAST = env("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
GROQ_MAIN = env("GROQ_MODEL_MAIN", "openai/gpt-oss-120b")
MISTRAL_MODEL = env("MISTRAL_MODEL_EVAL", "mistral-small-latest")
DEEPSEEK_MODEL = env("DEEPSEEK_MODEL_EVAL", "deepseek-flash")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def has_mistral() -> bool:
    return bool(env("MISTRAL_API_KEY"))


def has_deepseek() -> bool:
    return bool(env("DEEPSEEK_API_KEY"))


def independent_checker() -> tuple[str, str] | None:
    """A model from a DIFFERENT family than the gpt-oss models that write the
    questions and answer them. DeepSeek first, then Mistral; None if neither key is set."""
    if has_deepseek():
        return "deepseek", DEEPSEEK_MODEL
    if has_mistral():
        return "mistral", MISTRAL_MODEL
    return None


class _Cache:
    def __init__(self, path):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k):
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k, v):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self._db.commit()


_cache = _Cache(CACHE_DIR / "tools_llm.sqlite")
_last_call: dict[str, float] = {}


def _client(provider: str, model: str, temperature: float, json_mode: bool):
    if provider == "groq":
        from langchain_groq import ChatGroq
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        return ChatGroq(model=model, api_key=env("GROQ_API_KEY"), temperature=temperature,
                        model_kwargs=kwargs, max_retries=0)
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        return ChatMistralAI(model=model, api_key=env("MISTRAL_API_KEY"),
                             temperature=temperature, model_kwargs=kwargs, max_retries=0)
    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        return ChatOpenAI(model=model, api_key=env("DEEPSEEK_API_KEY"), base_url=DEEPSEEK_BASE_URL,
                          temperature=temperature, model_kwargs=kwargs, max_retries=0, timeout=120)
    raise ValueError(provider)


def complete(prompt: str, *, provider: str, model: str, temperature: float = 0.0,
             json_mode: bool = True, min_interval: float = 2.0, max_attempts: int = 8,
             cache_tag: str = "") -> str:
    key = hashlib.sha256(
        json.dumps([provider, model, temperature, json_mode, cache_tag, prompt]).encode()
    ).hexdigest()
    hit = _cache.get(key)
    if hit is not None:
        return hit

    llm = _client(provider, model, temperature, json_mode)
    delay = 5.0
    for attempt in range(1, max_attempts + 1):
        wait = min_interval - (time.time() - _last_call.get(provider, 0))
        if wait > 0:
            time.sleep(wait)
        _last_call[provider] = time.time()
        try:
            text = llm.invoke(prompt).content
            _cache.put(key, text)
            return text
        except Exception as e:  # rate limits, timeouts, transient 5xx
            msg = str(e)
            if attempt == max_attempts:
                raise
            transient = any(s in msg for s in ("429", "rate", "Rate", "timeout", "503", "502", "500"))
            if not transient and "json_validate_failed" not in msg:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 90)
    raise RuntimeError("unreachable")


def parse_json(text: str):
    """Pull the first JSON object/array out of a model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(m.group(1))


def complete_json(prompt: str, **kw):
    return parse_json(complete(prompt, json_mode=True, **kw))
