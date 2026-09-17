"""Google Gemini - free tier, no card required.

Get a key at https://aistudio.google.com/apikey and put it in .env as
GOOGLE_API_KEY.

Free-tier models return 503 under load fairly often, so a request that fails
before producing any text is retried on the next model in the fallback list.
"""

import json
import os
import re
import time
from typing import Iterator

from backend.providers.base import Provider, ProviderError, ProviderNotReady
from utils import config

ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")

# Codes worth retrying on a different model: overloaded, or rate limited.
RETRYABLE_CODES = {429, 503}


class _ModelUnavailable(Exception):
    """Internal: this model is busy or over quota; another one might work."""

    def __init__(self, model: str, cause: Exception):
        self.model = model
        self.cause = cause
        self.code = getattr(cause, "code", None)
        self.retry_after = retry_delay_seconds(cause)


def retry_delay_seconds(exc) -> float | None:
    """How long the API asked us to wait, if it said.

    A 429 carries a google.rpc.RetryInfo block and also spells the delay out
    in the message ("Please retry in 27.39s"). Honouring it beats guessing.
    """
    details = getattr(exc, "details", None)
    if isinstance(details, (dict, list)):
        blob = json.dumps(details)
    else:
        blob = str(details or "")
    blob += " " + str(getattr(exc, "message", "") or "")

    match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', blob)
    if match:
        return float(match.group(1))
    match = re.search(r"retry in ([0-9.]+)s", blob)
    if match:
        return float(match.group(1))
    return None


def quota_limit(exc) -> str | None:
    """The quota that was exceeded, e.g. '20 requests/day per model'.

    The QuotaFailure block names the quota id, which says whether the window
    is per-day or per-minute - worth getting right, because the advice to the
    user is completely different.
    """
    details = getattr(exc, "details", None)
    blob = json.dumps(details) if isinstance(details, (dict, list)) else str(details or "")
    blob += " " + str(getattr(exc, "message", "") or "")

    value = re.search(r'"quotaValue"\s*:\s*"?(\d+)"?', blob) or re.search(r"limit:\s*(\d+)", blob)
    if not value:
        return None

    quota_id = re.search(r'"quotaId"\s*:\s*"([^"]+)"', blob)
    window = "request"
    if quota_id:
        if "PerDay" in quota_id.group(1):
            window = "requests/day"
        elif "PerMinute" in quota_id.group(1):
            window = "requests/min"
    scope = " per model" if quota_id and "PerModel" in quota_id.group(1) else ""
    return f"{value.group(1)} {window}{scope}"


class GeminiProvider(Provider):
    name = "gemini"
    note = "Free tier. Your documents are sent to Google."
    key_variable = "GOOGLE_API_KEY"

    def __init__(self, model: str = config.GEMINI_MODEL, api_key: str | None = None):
        self.model = model
        # A key passed in wins over the environment: it came from someone
        # typing it into this session, which is a deliberate override of
        # whatever .env holds.
        self.api_key = (api_key or "").strip() or None

    def _api_key(self) -> str:
        if self.api_key:
            return self.api_key
        for var in ENV_VARS:
            key = os.environ.get(var)
            if key:
                return key
        raise ProviderNotReady(
            "No Gemini API key found. Get a free one at "
            "https://aistudio.google.com/apikey, then put it in your .env file "
            "as GOOGLE_API_KEY=... and restart the app."
        )

    def check_ready(self) -> None:
        self._api_key()

    def _client(self):
        from google import genai

        return genai.Client(api_key=self._api_key())

    def list_models(self) -> list[str]:
        """What this key can actually use - handy when a model name is rejected."""
        client = self._client()
        names = []
        for model in client.models.list():
            actions = getattr(model, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append((model.name or "").removeprefix("models/"))
        return sorted(n for n in names if n)

    def candidates(self) -> list[str]:
        """The configured model first, then the fallbacks, without duplicates."""
        ordered = [self.model, *config.GEMINI_FALLBACK_MODELS]
        seen, result = set(), []
        for name in ordered:
            if name and name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Try each model in turn; if all are busy, wait and go round again.

        Free-tier 503s and per-minute rate limits are usually over in seconds,
        so a couple of rounds with a short backoff turns most of them into a
        slightly slow answer rather than a failed question.
        """
        busy: list[str] = []
        last: _ModelUnavailable | None = None

        for round_number in range(config.GEMINI_RETRY_ROUNDS):
            if round_number:
                # Prefer the delay the API asked for over an invented backoff.
                asked = last.retry_after if last else None
                wait = min(asked or config.GEMINI_RETRY_BACKOFF * round_number,
                           config.GEMINI_MAX_WAIT)
                time.sleep(wait)

            for model in self.candidates():
                try:
                    yield from self._stream_one(model, system, prompt)
                    return
                except _ModelUnavailable as exc:
                    # Nothing was emitted yet, so it is safe to try another.
                    busy.append(exc.model)
                    last = exc

        tried = ", ".join(dict.fromkeys(busy))
        if last is not None and last.code == 429:
            limit = quota_limit(last.cause)
            cap = f" The free tier allows {limit}." if limit else ""
            raise ProviderError(
                f"Gemini free-tier quota exceeded on: {tried}.{cap} "
                "The quota is per model, so adding another model to "
                "GEMINI_FALLBACK_MODELS in utils/config.py buys more headroom. "
                "Otherwise switch the provider in the sidebar, or wait for the "
                "quota to reset."
            )
        raise ProviderError(
            f"Every Gemini model is overloaded right now ({tried}). This is "
            "Google's side, not yours - try again shortly, or switch the "
            "provider in the sidebar."
        )

    def _stream_one(self, model: str, system: str, prompt: str) -> Iterator[str]:
        from google.genai import errors, types

        client = self._client()
        yielded = False
        try:
            responses = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=config.GEMINI_MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                    # We pass no tools, so turn off the SDK's function-calling
                    # machinery - it only emits a warning here.
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            for response in responses:
                text = response.text
                if text:
                    yielded = True
                    yield text
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if not yielded and code in RETRYABLE_CODES:
                raise _ModelUnavailable(model, exc) from exc
            raise ProviderError(_explain(exc, model)) from exc

        if not yielded:
            raise ProviderError(
                "Gemini returned an empty response. This usually means the "
                "request was blocked by a safety filter - try rephrasing."
            )


def _explain(exc, model: str) -> str:
    """Turn the common API errors into something actionable."""
    message = getattr(exc, "message", None) or str(exc)
    code = getattr(exc, "code", None)

    if code == 404:
        return (
            f"Gemini rejected the model name {model!r} - it may have been "
            f"retired ({message}). Run `.venv/Scripts/python.exe -m utils.check "
            "--models` to list what your key can use, then set GEMINI_MODEL in "
            "utils/config.py."
        )
    if code in (401, 403):
        return f"Gemini rejected the API key: {message}"
    return f"Gemini request failed: {message}"
