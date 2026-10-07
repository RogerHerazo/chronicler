from chronicler.config import Settings, get_secret
from chronicler.llm.base import LLMProvider


def make_provider(settings: Settings) -> LLMProvider:
    if settings.provider == "ollama":
        from chronicler.llm.ollama import OllamaProvider

        return OllamaProvider(settings.ollama_url, settings.ollama_model)
    from chronicler.llm.claude import ClaudeProvider

    return ClaudeProvider(get_secret("anthropic_api_key"), settings.claude_model)
