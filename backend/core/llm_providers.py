"""
LLM provider selection. One provider is active at a time (settings.LLM_PROVIDER);
there is deliberately no automatic fallback, so an answer never silently comes
from a different model than the one configured.

  groq        ChatGroq, openai/gpt-oss-120b (main) + gpt-oss-20b (fast)
  deepseek    ChatOpenAI against api.deepseek.com, deepseek-flash, thinking off
  cloudflare  ChatOpenAI against Workers AI's OpenAI-compatible endpoint
"""
from __future__ import annotations

from typing import Literal

from config import settings

PROVIDERS = ("groq", "deepseek", "cloudflare")


class ProviderConfigError(RuntimeError):
    pass


def active_provider() -> str:
    p = (settings.LLM_PROVIDER or "groq").strip().lower()
    if p not in PROVIDERS:
        raise ProviderConfigError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {p!r}")
    return p


def model_name(role: Literal["main", "fast"], provider: str | None = None) -> str:
    p = provider or active_provider()
    return getattr(settings, f"{p.upper()}_MODEL_{role.upper()}")


def describe() -> dict[str, str]:
    """What the app is running on (shown by /health and recorded by the eval harness)."""
    p = active_provider()
    return {"provider": p, "main": model_name("main", p), "fast": model_name("fast", p)}


def make_llm(role: Literal["main", "fast"], temperature: float, streaming: bool = False,
             provider: str | None = None):
    p = provider or active_provider()
    model = model_name(role, p)

    if p == "groq":
        from langchain_groq import ChatGroq
        if not settings.GROQ_API_KEY:
            raise ProviderConfigError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        return ChatGroq(model=model, api_key=settings.GROQ_API_KEY,
                        temperature=temperature, streaming=streaming)

    from langchain_openai import ChatOpenAI

    if p == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise ProviderConfigError("LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set")
        return ChatOpenAI(model=model, api_key=settings.DEEPSEEK_API_KEY,
                          base_url=settings.DEEPSEEK_BASE_URL, temperature=temperature,
                          streaming=streaming, stream_usage=True, max_retries=2, timeout=120,
                          extra_body={"thinking": {"type": "disabled"}})

    # cloudflare
    if not (settings.CLOUDFLARE_ACCOUNT_ID and settings.CLOUDFLARE_API_TOKEN):
        raise ProviderConfigError(
            "LLM_PROVIDER=cloudflare but CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN are not set")
    base = f"https://api.cloudflare.com/client/v4/accounts/{settings.CLOUDFLARE_ACCOUNT_ID}/ai/v1"
    return ChatOpenAI(model=model, api_key=settings.CLOUDFLARE_API_TOKEN, base_url=base,
                      temperature=temperature, streaming=streaming, stream_usage=True,
                      max_retries=2, timeout=120)
