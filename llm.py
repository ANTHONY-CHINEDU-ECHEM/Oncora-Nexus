"""Local language model access through Ollama.

Oncora Nexus never sends text to a hosted API. The default model is
Llama 3.1 8B Instruct served by Ollama on the same host or inside the Docker
network; any Ollama model tag can be selected with ONCORA_LLM_MODEL (for
example qwen2.5:7b, mistral:7b or a medically tuned model such as
meditron or a local BioMistral build).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import httpx

from oncora.config import settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        cfg = settings.llm
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.model = model or cfg.model
        self.cfg = cfg

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            if r.status_code != 200:
                return False
            names = {m.get("name", "") for m in r.json().get("models", [])}
            return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)
        except Exception:
            return False

    def _payload(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
                "num_ctx": 8192,
            },
        }

    def chat(self, messages: list[dict]) -> str:
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=self._payload(messages, False), timeout=self.cfg.timeout_seconds)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc

    def stream(self, messages: list[dict]) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST", f"{self.base_url}/api/chat", json=self._payload(messages, True), timeout=self.cfg.timeout_seconds
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
