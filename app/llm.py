from abc import ABC, abstractmethod

import anthropic

from .config import Settings


class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, system: str, user: str) -> str: ...


class AnthropicBackend(LLMBackend):
    def __init__(self, settings: Settings) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    async def generate(self, system: str, user: str) -> str:
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            response = await stream.get_final_message()
        return next(b.text for b in response.content if b.type == "text")


class LLMClient:
    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    async def generate(self, system: str, user: str) -> str:
        return await self._backend.generate(system, user)


def create_llm_client(settings: Settings) -> LLMClient:
    backends: dict[str, type[LLMBackend]] = {
        "anthropic": AnthropicBackend,
    }
    cls = backends.get(settings.llm_provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
    return LLMClient(cls(settings))
