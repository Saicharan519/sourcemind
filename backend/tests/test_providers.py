"""Provider selection is explicit, validated, and never falls back silently."""
import pytest

from config import settings
from core import llm_providers as lp


@pytest.fixture
def provider(monkeypatch):
    def use(name, **keys):
        monkeypatch.setattr(settings, "LLM_PROVIDER", name)
        for k, v in keys.items():
            monkeypatch.setattr(settings, k, v)
    return use


def test_groq_is_default_and_uses_gpt_oss(provider):
    provider("groq", GROQ_API_KEY="k")
    llm = lp.make_llm("main", 0.2)
    assert type(llm).__name__ == "ChatGroq" and llm.model_name == "openai/gpt-oss-120b"
    assert lp.describe() == {"provider": "groq", "main": "openai/gpt-oss-120b", "fast": "openai/gpt-oss-20b"}


def test_deepseek_uses_openai_client_with_thinking_off(provider):
    provider("deepseek", DEEPSEEK_API_KEY="k")
    llm = lp.make_llm("fast", 0.3, streaming=True)
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "deepseek-flash"
    assert llm.openai_api_base == "https://api.deepseek.com"
    assert llm.extra_body == {"thinking": {"type": "disabled"}}


def test_cloudflare_url_contains_account(provider):
    provider("cloudflare", CLOUDFLARE_ACCOUNT_ID="acc123", CLOUDFLARE_API_TOKEN="t")
    llm = lp.make_llm("main", 0.2)
    assert llm.openai_api_base.endswith("/accounts/acc123/ai/v1")
    assert llm.model_name == "@cf/openai/gpt-oss-120b"


@pytest.mark.parametrize("name,keys", [
    ("deepseek", {"DEEPSEEK_API_KEY": ""}),
    ("cloudflare", {"CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_API_TOKEN": "t"}),
    ("groq", {"GROQ_API_KEY": ""}),
])
def test_missing_key_is_a_clear_error_not_a_fallback(provider, name, keys):
    provider(name, **keys)
    with pytest.raises(lp.ProviderConfigError):
        lp.make_llm("main", 0.2)


def test_unknown_provider_rejected(provider):
    provider("openai")
    with pytest.raises(lp.ProviderConfigError):
        lp.active_provider()
