from __future__ import annotations

import json

import httpx
import pytest

from chronicler.llm import prompts
from chronicler.llm.base import CampaignContext, ChunkAnalysis, ChunkContext, LLMError
from chronicler.llm.claude import ClaudeProvider
from chronicler.llm.ollama import OllamaProvider

VALID = {
    "beats": ["The party arrived."],
    "recap": "They arrived.",
    "entities": [{"name": "Strahd", "kind": "npc", "in_notes": True, "note": "Watched."}],
    "threads": [{"title": "The letter", "change": "opened", "note": "Unsigned."}],
}


def ctx(**kw) -> ChunkContext:
    base = {
        "campaign": CampaignContext(
            "Curse of Strahd", "Strahd von Zarovich rules Barovia.", "English"
        ),
        "chunk_index": 1,
        "start_s": 900,
        "end_s": 1800,
        "transcript": "[0:15:01] hola",
    }
    return ChunkContext(**{**base, **kw})


def ollama_with(handler) -> OllamaProvider:
    provider = OllamaProvider("http://ollama.test", "qwen3:14b")
    provider._http = httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return provider


def test_prompts_include_context_and_language() -> None:
    c = ctx(
        previous_recap="Earlier recap.",
        known_entities=[("Ireena", "npc", ["Irina"])],
        open_threads=["The letter"],
    )
    system = prompts.system_prompt(c.campaign)
    user = prompts.chunk_user_prompt(c)
    assert "Write everything in English" in system
    assert "<campaign_notes>" in system and "Strahd von Zarovich" in system
    assert "- Ireena [npc] (also heard as: Irina)" in user
    assert "- The letter" in user
    assert "part 2 (0:15:00 to 0:30:00)" in user
    assert "Earlier recap." in user


def test_system_prompt_is_stable_for_caching() -> None:
    a = prompts.system_prompt(ctx().campaign)
    b = prompts.system_prompt(ctx(chunk_index=5, transcript="other").campaign)
    assert a == b


def test_ollama_parses_structured_output() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body)
        return httpx.Response(200, json={"message": {"content": json.dumps(VALID)}})

    result = ollama_with(handler).analyze_chunk(ctx())
    assert isinstance(result, ChunkAnalysis)
    assert result.entities[0].name == "Strahd"
    assert seen["format"]["title"] == "ChunkAnalysis"
    assert seen["messages"][0]["role"] == "system"


def test_ollama_retries_invalid_json_once() -> None:
    replies = iter(["not json", json.dumps(VALID)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": next(replies)}})

    assert ollama_with(handler).analyze_chunk(ctx()).recap == "They arrived."


def test_ollama_gives_up_after_two_invalid_replies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "{}"}})

    with pytest.raises(LLMError) as e:
        ollama_with(handler).analyze_chunk(ctx())
    assert e.value.retryable


def test_ollama_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
        return httpx.Response(404, json={"error": "model not found"})

    provider = ollama_with(handler)
    status = provider.check()
    assert not status.ok and "ollama pull qwen3:14b" in status.fix_hint
    with pytest.raises(LLMError, match="not pulled"):
        provider.analyze_chunk(ctx())


def test_ollama_not_running() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    status = ollama_with(handler).check()
    assert not status.ok and "not running" in status.detail


def test_claude_without_key() -> None:
    provider = ClaudeProvider(None)
    assert not provider.check().ok
    with pytest.raises(LLMError, match="No Anthropic API key"):
        provider.analyze_chunk(ctx())
