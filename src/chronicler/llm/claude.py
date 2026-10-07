"""Claude provider (Anthropic API)."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

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

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class ClaudeProvider:
    name = "claude"

    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL):
        self.model = model
        self._api_key = api_key
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=4) if api_key else None

    def _system(self, ctx: Any) -> list[dict[str, Any]]:
        # Stable across every call in a session, so it is written to the cache
        # once and read back cheaply for each later chunk.
        return [
            {
                "type": "text",
                "text": prompts.system_prompt(ctx.campaign),
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

    def _parse(self, system: list[dict[str, Any]], user: str, schema: type[T], effort: str) -> T:
        if self._client is None:
            raise LLMError("No Anthropic API key configured.")
        try:
            response = self._client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=system,  # pyright: ignore[reportArgumentType]
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": effort},  # pyright: ignore[reportArgumentType]
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("The Anthropic API key was rejected.") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"The API key cannot use {self.model}.") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Unknown Claude model '{self.model}'.") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"Claude rejected the request: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Rate limited by the Anthropic API.", retryable=True) from e
        except anthropic.APIStatusError as e:
            raise LLMError(
                f"Anthropic API error {e.status_code}: {e.message}",
                retryable=e.status_code >= 500,
            ) from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API.", retryable=True) from e

        usage = response.usage
        log.info(
            "claude %s: in=%s cache_read=%s cache_write=%s out=%s",
            schema.__name__,
            usage.input_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
            usage.output_tokens,
        )
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to analyze this part of the transcript.")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude's answer was cut off.", retryable=True)
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("Claude returned no structured output.", retryable=True)
        return parsed

    def analyze_chunk(self, ctx: ChunkContext) -> ChunkAnalysis:
        return self._parse(
            self._system(ctx), prompts.chunk_user_prompt(ctx), ChunkAnalysis, "medium"
        )

    def summarize_session(self, ctx: SessionContext) -> SessionSummary:
        return self._parse(
            self._system(ctx), prompts.summary_user_prompt(ctx), SessionSummary, "high"
        )

    def check(self) -> ProviderStatus:
        if self._client is None:
            return ProviderStatus(
                False,
                "No Anthropic API key configured.",
                "Add your key in Settings (stored in the OS keychain) or set ANTHROPIC_API_KEY. "
                "Get one at https://console.anthropic.com/settings/keys",
            )
        try:
            self._client.with_options(max_retries=1, timeout=15.0).models.retrieve(self.model)
        except anthropic.AuthenticationError:
            return ProviderStatus(
                False,
                "The Anthropic API key was rejected.",
                "Check or replace the key in Settings.",
            )
        except anthropic.NotFoundError:
            return ProviderStatus(
                False,
                f"Model '{self.model}' is not available to this key.",
                f"Pick another Claude model in Settings (default: {DEFAULT_MODEL}).",
            )
        except anthropic.APIConnectionError:
            return ProviderStatus(
                False, "Could not reach api.anthropic.com.", "Check your internet connection."
            )
        except anthropic.APIStatusError as e:
            return ProviderStatus(False, f"Anthropic API error {e.status_code}.", str(e.message))
        return ProviderStatus(True, f"API key works, model {self.model} is available.")
