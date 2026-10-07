# Adding an LLM provider

A provider turns transcript text into the structured models in `src/chronicler/llm/base.py`.

1. Create `src/chronicler/llm/<name>.py` with a class that has `name`, `model` and three methods:
   - `analyze_chunk(ctx: ChunkContext) -> ChunkAnalysis`
   - `summarize_session(ctx: SessionContext) -> SessionSummary`
   - `check() -> ProviderStatus`, a fast connectivity and configuration check used by the health page.
2. Build prompts with `prompts.system_prompt()`, `prompts.chunk_user_prompt()` and `prompts.summary_user_prompt()`, so every provider sees the same instructions.
3. Raise `LLMError` for failures, with `retryable=True` for transient ones (network, rate limits, malformed output).
   Messages are shown to users, so make them actionable.
4. Register the provider in `make_provider()` in `src/chronicler/llm/__init__.py`, add its settings to `config.Settings` and the Settings page.
5. Add tests with a mocked HTTP transport, like `tests/test_llm.py`.
