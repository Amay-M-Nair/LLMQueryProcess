"""Ollama - a model running on your own machine. Free, offline, private.

Install from https://ollama.com, then pull a model that fits your GPU:

    ollama pull llama3.2:3b

Uses the stdlib rather than another dependency: Ollama's chat endpoint streams
newline-delimited JSON, which is easy enough to read directly.
"""

import json
import urllib.error
import urllib.request
from typing import Iterator

from backend.providers.base import Provider, ProviderError, ProviderNotReady
from utils import config


class OllamaProvider(Provider):
    name = "ollama"
    note = "Free and fully offline. Small models cite less reliably."

    def __init__(self, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    def check_ready(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as response:
                tags = json.loads(response.read())
        except Exception as exc:
            raise ProviderNotReady(
                f"Can't reach Ollama at {self.host}. Install it from "
                "https://ollama.com and make sure it is running."
            ) from exc

        installed = {m.get("name", "") for m in tags.get("models", [])}
        # Ollama reports "llama3.2:3b"; accept a bare "llama3.2" too.
        if not any(
            name == self.model or name.split(":")[0] == self.model.split(":")[0]
            for name in installed
        ):
            raise ProviderNotReady(
                f"Ollama is running but {self.model!r} isn't installed. Run:\n\n"
                f"    ollama pull {self.model}"
            )

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        self.check_ready()

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {
                    "temperature": config.TEMPERATURE,
                    "num_predict": config.OLLAMA_NUM_PREDICT,
                },
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=config.OLLAMA_TIMEOUT) as response:
                for line in response:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise ProviderError(f"Ollama error: {event['error']}")
                    piece = event.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if event.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
