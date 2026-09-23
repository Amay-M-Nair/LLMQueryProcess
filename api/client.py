"""How the UI reaches the pipeline: in this process, or over HTTP.

Both implement the same three operations, so app.py does not know which it
has. That is the point of the split - the UI stops importing the pipeline and
starts depending on an interface, which is what lets the service move to
another machine without the page noticing.

Local is the default, because one process is easier to run than two and
nothing about a single-user app needs a network hop. Set API_URL in
utils/config.py to point the page at a remote service instead. No such
service ships here any more - the HTTP half is the seam, kept because it is
what stops app.py importing the pipeline directly.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from utils import config, logs

log = logs.get(__name__)


@dataclass
class IngestResult:
    indexed: list[str]
    reindexed: list[str]
    skipped: list[tuple[str, str]]
    chunks_added: int
    chunks: int

    @property
    def changed(self) -> bool:
        return bool(self.indexed or self.reindexed)


class Client(Protocol):
    """What the UI needs, and nothing else."""

    def describe(self, collection: str) -> dict: ...
    def add_documents(self, collection: str, files, on_progress=None) -> IngestResult: ...
    def clear(self, collection: str) -> None: ...
    def ask(self, collection, question, history, length,
            provider_name=None, api_key=None, on_stage=None) -> Iterator[dict]: ...


class LocalClient:
    """Calls the pipeline in this process. No server, no network."""

    def describe(self, collection: str) -> dict:
        from api import collections

        return collections.describe(collection)

    def add_documents(self, collection, files, on_progress=None) -> IngestResult:
        from api import collections

        paths = collections.save_uploads(collection, files)
        _, report = collections.ingest(collection, paths, on_progress=on_progress)
        return IngestResult(
            report.indexed, report.reindexed, report.skipped,
            report.chunks_added, report.total_chunks,
        )

    def clear(self, collection: str) -> None:
        from api import collections

        collections.clear(collection)

    def ask(self, collection, question, history, length,
            provider_name=None, api_key=None, on_stage=None) -> Iterator[dict]:
        from api import collections
        from backend.llm import get_provider
        from backend.query_processor import process

        store = collections.load(collection)
        # Both arguments matter and neither can be guessed from config:
        # the provider is whichever the sidebar has selected, and the key
        # may have been typed into the page rather than set in .env. A
        # deployed copy has no .env at all, so dropping the key here left
        # the sidebar reporting Ready and every question failing.
        provider = get_provider(provider_name or config.PROVIDER, api_key=api_key or None)

        stages: list[str] = []
        plan = process(
            question, store=store, history=history, provider=provider,
            on_stage=stages.append, length=length,
        )
        for stage in stages:
            yield {"type": "stage", "text": stage}

        trace = plan.trace
        yield {
            "type": "route",
            "intent": trace.intent.name, "method": trace.intent.method,
            "intent_reason": trace.intent.reason,
            "route": trace.route.name, "route_reason": trace.route.reason,
            "searched_for": (
                trace.rewrite.query if (trace.rewrite and trace.rewrite.changed) else None
            ),
            "api_calls": trace.api_calls,
            "timings": {k: round(v * 1000) for k, v in trace.timings.items()},
            "notes": trace.notes,
        }

        if plan.answer is not None or plan.message is not None:
            yield {"type": "token",
                   "text": plan.answer if plan.answer is not None else plan.message}
        else:
            for piece in plan.stream():
                yield {"type": "token", "text": piece}

        yield {
            "type": "sources",
            "sources": [
                {"label": c.label, "source": c.source, "page": c.page,
                 "score": round(s, 3), "text": c.text}
                for c, s in plan.sources
            ],
        }


class HttpClient:
    """Calls a remote service over HTTP.

    The ask() stream is read line by line rather than waited for: the whole
    reason the service emits newline-delimited JSON is so the page can show
    the answer arriving, and buffering it here would throw that away.
    """

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self):
        import httpx

        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def describe(self, collection: str) -> dict:
        with self._client() as http:
            response = http.get(f"/collections/{collection}")
            response.raise_for_status()
            return response.json()

    def add_documents(self, collection, files, on_progress=None) -> IngestResult:
        # Progress arrives only at the end over HTTP: the upload is one
        # request and the service cannot report back into it mid-flight.
        if on_progress:
            on_progress("Uploading and indexing")

        payload = [("files", (name, data)) for name, data in files]
        with self._client() as http:
            response = http.post(f"/collections/{collection}/documents", files=payload)
            response.raise_for_status()
            body = response.json()

        return IngestResult(
            body["indexed"], body["reindexed"],
            [(s["name"], s["reason"]) for s in body["skipped"]],
            body["chunks_added"], body["chunks"],
        )

    def clear(self, collection: str) -> None:
        with self._client() as http:
            http.delete(f"/collections/{collection}").raise_for_status()

    def ask(self, collection, question, history, length,
            provider_name=None, api_key=None, on_stage=None) -> Iterator[dict]:
        # api_key is accepted and never sent. A key typed into this page
        # belongs to whoever typed it; handing it to a remote service
        # would disclose it to a third party. A service answers with its
        # own key and its own choice of provider.
        form = {
            "question": question,
            "history": json.dumps(history or []),
            "length": length,
        }
        with self._client() as http:
            with http.stream("POST", f"/collections/{collection}/ask", data=form) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.strip():
                        yield json.loads(line)


def get_client() -> Client:
    """Whichever the configuration asks for."""
    if config.API_URL:
        log.info("using the API at %s", config.API_URL)
        return HttpClient(config.API_URL)
    return LocalClient()
