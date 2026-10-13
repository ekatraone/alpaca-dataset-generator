from types import SimpleNamespace

import pytest

import llm_backend
from llm_backend import LLMBackend


class FakeOpenAI:
    """Stands in for openai.OpenAI; records calls and replays scripted responses."""

    instances = []

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.calls = []
        self.script = []  # list of either str (reply) or Exception
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        FakeOpenAI.instances.append(self)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.instances.clear()
    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_backend.time, "sleep", lambda s: None)
    return FakeOpenAI


def test_api_mode_requires_key(monkeypatch, fake_openai):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        LLMBackend(mode="api")


def test_api_mode_reads_env_key_and_defaults(monkeypatch, fake_openai, config):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    b = LLMBackend(mode="api")
    client = fake_openai.instances[-1]
    assert client.api_key == "sk-or-test"
    assert b.base_url == config["api_base_url"]
    assert b.model_name == config["api_model"]
    assert b.is_api()


def test_api_generate_sends_prompt_and_strips(fake_openai, config):
    b = LLMBackend(mode="api", api_key="k", model="m/x", base_url="http://localhost:1/v1")
    client = fake_openai.instances[-1]
    client.script = ["  hello world  "]
    assert b.generate("prompt text", max_tokens=42) == "hello world"
    call = client.calls[0]
    assert call["model"] == "m/x"
    assert call["max_tokens"] == 42
    assert call["messages"] == [{"role": "user", "content": "prompt text"}]
    assert call["temperature"] == config["api_temperature"]


def test_api_generate_retries_then_succeeds(fake_openai, config):
    config["api_max_retries"] = 3
    b = LLMBackend(mode="api", api_key="k")
    client = fake_openai.instances[-1]
    client.script = [RuntimeError("429"), RuntimeError("500"), "ok"]
    assert b.generate("p") == "ok"
    assert len(client.calls) == 3


def test_api_generate_gives_up_after_max_retries(fake_openai, config):
    config["api_max_retries"] = 2
    b = LLMBackend(mode="api", api_key="k")
    client = fake_openai.instances[-1]
    client.script = [RuntimeError("down"), RuntimeError("down")]
    assert b.generate("p") == ""
    assert len(client.calls) == 2


def test_api_sentiment_prompt_includes_text(fake_openai):
    b = LLMBackend(mode="api", api_key="k")
    client = fake_openai.instances[-1]
    client.script = ["Negative. It complains."]
    out = b.generate_sentiment("the service was awful")
    assert out.startswith("Negative")
    assert "the service was awful" in client.calls[0]["messages"][0]["content"]


def test_local_mode_requires_models():
    with pytest.raises(ValueError):
        LLMBackend(mode="local")


def test_invalid_mode():
    with pytest.raises(ValueError):
        LLMBackend(mode="cloud")
