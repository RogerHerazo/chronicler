"""Ollama provider: free and fully local analysis."""

from __future__ import annotations

import json
import logging
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from chronicler.llm import prompts
from chronicler.llm.base import (
    ChunkAnalysis,
    ChunkContext,
    LLMError,
    ProviderStatus,
    SessionContext,
    SessionSummary,
)

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:14b"
CONTEXT_TOKENS = 32_768

T = TypeVar("T", bound=BaseModel)


class OllamaProvider:
    name = "ollama"

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.model = model
        self._http = httpx.Client(base_url=self.url, timeout=timeout)

    def _chat(self, system: str, user: str, schema: type[T]) -> T:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"num_ctx": CONTEXT_TOKENS, "temperature": 0.2},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_error = ""
        # Small local models occasionally produce JSON that misses the schema;
        # one retry usually fixes it.
        for _attempt in range(2):
            try:
                r = self._http.post("/api/chat", json=payload)
            except httpx.TransportError as e:
                raise LLMError(f"Could not reach Ollama at {self.url}.", retryable=True) from e
            if r.status_code == 404:
                raise LLMError(f"Ollama model '{self.model}' is not pulled.")
            if r.status_code >= 400:
                raise LLMError(
                    f"Ollama error {r.status_code}: {r.text[:200]}", retryable=r.status_code >= 500
                )
            content = r.json().get("message", {}).get("content", "")
            try:
                return schema.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                log.warning("ollama returned invalid JSON, retrying: %s", last_error[:200])
        raise LLMError(f"Ollama returned invalid output: {last_error[:200]}", retryable=True)

    def analyze_chunk(self, ctx: ChunkContext) -> ChunkAnalysis:
        return self._chat(
            prompts.system_prompt(ctx.campaign), prompts.chunk_user_prompt(ctx), ChunkAnalysis
        )

    def summarize_session(self, ctx: SessionContext) -> SessionSummary:
        return self._chat(
            prompts.system_prompt(ctx.campaign), prompts.summary_user_prompt(ctx), SessionSummary
        )

    def check(self) -> ProviderStatus:
        try:
            r = self._http.get("/api/tags", timeout=5.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return ProviderStatus(
                False,
                f"Ollama is not running at {self.url}.",
                "Install Ollama from https://ollama.com and start it, or switch to Claude in "
                "Settings.",
            )
        names = {m.get("name", "") for m in r.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            return ProviderStatus(
                False,
                f"Model '{self.model}' is not pulled.",
                f"Run: ollama pull {self.model}",
            )
        return ProviderStatus(True, f"Ollama is running and {self.model} is available.")
