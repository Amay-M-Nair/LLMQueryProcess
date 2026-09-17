"""The contract every answer provider implements.

A provider takes a system prompt and a user prompt, and yields the answer as
text arrives. That is the whole interface - everything else (retrieval,
chunking, the UI) is provider-independent.
"""

from abc import ABC, abstractmethod
from typing import Iterator


class ProviderNotReady(RuntimeError):
    """Provider can't run yet - missing key, server not started, etc.

    The message is shown directly to the user, so it should say what to do.
    """


class ProviderError(RuntimeError):
    """The provider was reachable but the request failed."""


class Provider(ABC):
    name: str = "provider"
    #: Human-readable note shown in the UI (cost, privacy, etc.)
    note: str = ""

    @abstractmethod
    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield the answer in pieces as it is generated."""

    @abstractmethod
    def check_ready(self) -> None:
        """Raise ProviderNotReady if this provider cannot run right now."""

    def is_ready(self) -> bool:
        try:
            self.check_ready()
            return True
        except ProviderNotReady:
            return False
