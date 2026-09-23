"""Azriel over HTTP.

Run it with:

    .venv/Scripts/python.exe -m uvicorn api.main:app --reload

The routes decide nothing. What happens to a question is decided in
backend/query_processor.py, exactly as it is for the Streamlit app - this
layer resolves the collection, hands the question over, and streams back what
the orchestrator produced.

Answers arrive as newline-delimited JSON rather than plain text, because a
caller needs the routing and the sources alongside the prose. Returning only
prose would mean a second request that had to re-run the question to find out
what the first one did.
"""

import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Path as PathParam, UploadFile
from fastapi.responses import StreamingResponse

from api import collections
from api.collections import BadCollectionName
from backend.llm import PROVIDERS, ProviderError, ProviderNotReady, get_provider, model_for
from backend.query_processor import process
from utils import config, logs, prompts

log = logs.get(__name__)

app = FastAPI(
    title="Azriel",
    version="1.0",
    summary="Ask questions about your own documents, with answers cited back to the page.",
    description=(
        "Documents are kept in **collections**. A collection is a name you "
        "choose and keep sending; each one has its own index, and no "
        "collection can retrieve another's documents.\n\n"
        "There is no login. Whoever deploys this decides whether collection "
        "names are secret, and should put something in front if they must be."
    ),
)

Collection = Annotated[
    str,
    PathParam(description="Your collection name. Letters, digits, dashes, underscores."),
]


def _name(collection: str) -> str:
    try:
        return collections.normalise(collection)
    except BadCollectionName as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health", tags=["status"])
def health():
    """Whether the service can answer at all, and which model would do it."""
    provider = get_provider(config.PROVIDER)
    try:
        provider.check_ready()
        ready, detail = True, None
    except ProviderNotReady as exc:
        ready, detail = False, str(exc)

    return {
        "ready": ready,
        "provider": provider.name,
        "model": model_for(provider.name),
        "providers": list(PROVIDERS),
        "detail": detail,
        "lengths": list(prompts.LENGTHS),
    }


@app.get("/collections/{collection}", tags=["documents"])
def describe(collection: Collection):
    """What is indexed in this collection."""
    return collections.describe(_name(collection))


@app.post("/collections/{collection}/documents", tags=["documents"])
async def add_documents(collection: Collection, files: list[UploadFile] = File(...)):
    """Index one or more documents into this collection.

    Re-sending a file whose contents are unchanged is a no-op; re-sending one
    that has changed replaces the chunks held under that name, rather than
    leaving a stale copy to be retrieved later.
    """
    name = _name(collection)
    if not files:
        raise HTTPException(status_code=422, detail="No files were sent.")

    payload = [(f.filename or "unnamed", await f.read()) for f in files]
    try:
        paths = collections.save_uploads(name, payload)
    except BadCollectionName as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        store, report = collections.ingest(name, paths)
    except Exception as exc:
        log.warning("indexing failed for %s", name, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    return {
        "collection": name,
        "indexed": report.indexed,
        "reindexed": report.reindexed,
        "skipped": [{"name": n, "reason": r} for n, r in report.skipped],
        "chunks_added": report.chunks_added,
        "chunks": len(store),
    }


@app.delete("/collections/{collection}", tags=["documents"])
def clear(collection: Collection):
    """Delete this collection's index and the documents behind it."""
    name = _name(collection)
    return {"collection": name, "existed": collections.clear(name)}


@app.post("/collections/{collection}/ask", tags=["answering"])
def ask(
    collection: Collection,
    question: Annotated[str, Form(description="What you want to know.")],
    history: Annotated[str, Form(description="Prior turns as a JSON array.")] = "[]",
    length: Annotated[str, Form(description="Brief, Standard or Thorough.")] = prompts.DEFAULT_LENGTH,
):
    """Answer a question, streaming newline-delimited JSON.

    The stream carries four kinds of object: `stage` while the route is being
    worked out, one `route` describing what was decided, many `token` pieces
    of the answer, and finally `sources`.
    """
    name = _name(collection)

    try:
        turns = json.loads(history or "[]")
        if not isinstance(turns, list):
            raise ValueError
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="history must be a JSON array of turns."
        ) from exc

    store = collections.load(name)
    provider = get_provider(config.PROVIDER)

    def events():
        stages: list[str] = []
        try:
            plan = process(
                question, store=store, history=turns, provider=provider,
                on_stage=stages.append, length=length,
            )
        except Exception as exc:
            log.warning("could not plan a question for %s", name, exc_info=True)
            yield json.dumps({"type": "error", "text": str(exc)}) + "\n"
            return

        for stage in stages:
            yield json.dumps({"type": "stage", "text": stage}) + "\n"

        trace = plan.trace
        yield json.dumps({
            "type": "route",
            "intent": trace.intent.name,
            "method": trace.intent.method,
            "intent_reason": trace.intent.reason,
            "route": trace.route.name,
            "route_reason": trace.route.reason,
            "searched_for": (
                trace.rewrite.query if (trace.rewrite and trace.rewrite.changed) else None
            ),
            "api_calls": trace.api_calls,
            "timings": {k: round(v * 1000) for k, v in trace.timings.items()},
            "notes": trace.notes,
        }) + "\n"

        if plan.answer is not None or plan.message is not None:
            text = plan.answer if plan.answer is not None else plan.message
            yield json.dumps({"type": "token", "text": text}) + "\n"
        else:
            try:
                for piece in plan.stream():
                    yield json.dumps({"type": "token", "text": piece}) + "\n"
            except (ProviderNotReady, ProviderError) as exc:
                yield json.dumps({"type": "error", "text": str(exc)}) + "\n"
                return
            except Exception as exc:
                log.warning("generation failed for %s", name, exc_info=True)
                yield json.dumps({"type": "error", "text": f"The request failed: {exc}"}) + "\n"
                return

        yield json.dumps({
            "type": "sources",
            "sources": [
                {"label": chunk.label, "source": chunk.source, "page": chunk.page,
                 "score": round(score, 3), "text": chunk.text}
                for chunk, score in plan.sources
            ],
        }) + "\n"

    response = StreamingResponse(events(), media_type="application/x-ndjson")
    # Without this a proxy may hold the whole reply and defeat the streaming.
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    return response
