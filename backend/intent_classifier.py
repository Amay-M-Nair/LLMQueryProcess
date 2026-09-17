"""Work out what kind of question this is.

The order matters more than the cleverness. Most questions are settled by a
rule that costs nothing, and only the genuinely ambiguous ones reach the
model - which is the difference between one API call per question and three,
and on a free tier that is the difference between working all day and being
locked out by lunchtime.

Every path returns an Intent. There is no failure mode where classification
raises and takes the question down with it: a model that is unreachable, or
that replies with something other than JSON, falls back to a route that still
answers.
"""

import json
import re
from dataclasses import dataclass

from backend import calculator
from backend.llm import Provider, ProviderError, ProviderNotReady
from utils import prompts

GENERAL = "general"
CALCULATION = "calculation"
DOCUMENT_QUERY = "document_query"
SUMMARIZATION = "summarization"
UNKNOWN = "unknown"

INTENTS = (GENERAL, CALCULATION, DOCUMENT_QUERY, SUMMARIZATION, UNKNOWN)
NEEDS_RETRIEVAL = {DOCUMENT_QUERY, SUMMARIZATION}

SUMMARY_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+)?"
    r"(?:summari[sz]e|summari[sz]ation|tl;?dr|give me (?:a |an )?(?:summary|overview)|"
    r"what(?:'s| is) this (?:document|file|pdf) about|what do(?:es)? (?:this|these) "
    r"(?:document|file|pdf)s? (?:say|cover)|outline)\b",
    re.IGNORECASE,
)


@dataclass
class Intent:
    """What the query is, and how that was decided.

    `method` and `reason` are not decoration - they are what makes a wrong
    route debuggable, and the evaluation harness reports accuracy split by
    method to show whether the heuristics are pulling their weight.
    """

    name: str
    requires_retrieval: bool
    method: str   # heuristic | llm | fallback
    reason: str

    @property
    def from_model(self) -> bool:
        return self.method == "llm"


def _intent(name: str, method: str, reason: str, requires_retrieval=None) -> Intent:
    if requires_retrieval is None:
        requires_retrieval = name in NEEDS_RETRIEVAL
    return Intent(name, requires_retrieval, method, reason)


def classify_by_rule(query: str, has_index: bool) -> Intent | None:
    """The free pass. None means this one needs the model."""
    if not query.strip():
        return _intent(UNKNOWN, "heuristic", "the query is empty")

    if calculator.looks_arithmetic(query):
        return _intent(CALCULATION, "heuristic", "parses as an arithmetic expression")

    if SUMMARY_PATTERN.match(query):
        if not has_index:
            return _intent(
                UNKNOWN, "heuristic", "asks for a summary, but nothing is indexed"
            )
        return _intent(SUMMARIZATION, "heuristic", "asks for a summary")

    if not has_index:
        # Retrieval is impossible, so the only question worth asking is
        # whether the model can answer unaided - and it may as well try.
        return _intent(GENERAL, "heuristic", "no documents are indexed")

    return None


def parse_reply(reply: str) -> tuple[str, bool] | None:
    """Pull (intent, requires_retrieval) out of the model's reply.

    Models wrap JSON in code fences, prefix it with "Here you go:", or return
    the right object inside a larger one. Finding the first balanced object
    and reading it is more forgiving than json.loads on the whole string, and
    costs nothing when the reply was clean to begin with.
    """
    if not reply:
        return None

    candidate = reply.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    match = re.search(r"\{.*?\}", candidate, re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    name = str(payload.get("intent", "")).strip().lower()
    if name not in INTENTS:
        return None

    retrieval = payload.get("requires_retrieval")
    if not isinstance(retrieval, bool):
        # The model named an intent but garbled the flag; the intent is the
        # part that took judgement, so keep it and derive the rest.
        retrieval = name in NEEDS_RETRIEVAL

    return name, retrieval


def classify(
    query: str,
    has_index: bool,
    sources: list[str] | None = None,
    provider: Provider | None = None,
) -> Intent:
    """Name the intent, using the model only when the rules cannot decide."""
    ruled = classify_by_rule(query, has_index)
    if ruled is not None:
        return ruled

    if provider is None:
        return _fallback(has_index, "no provider available to classify with")

    system, user = prompts.intent_prompt(query, sources or [])
    try:
        reply = provider.complete(system, user)
    except (ProviderNotReady, ProviderError) as exc:
        return _fallback(has_index, f"classifier call failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - never let this sink the question
        return _fallback(has_index, f"classifier call failed: {exc}")

    parsed = parse_reply(reply)
    if parsed is None:
        return _fallback(has_index, "classifier did not return usable JSON")

    name, retrieval = parsed
    if retrieval and not has_index:
        # The model cannot know what is indexed better than we do.
        return _intent(name, "llm", "model chose retrieval, but nothing is indexed",
                       requires_retrieval=False)

    return _intent(name, "llm", "classified by the model", requires_retrieval=retrieval)


def _fallback(has_index: bool, reason: str) -> Intent:
    """Where a question goes when classification could not be done.

    With documents indexed the safe default is to retrieve: an answer grounded
    in the user's files and cited is recoverable if the routing was wrong,
    whereas answering from memory when the files held the answer looks
    authoritative and is not.
    """
    if has_index:
        return _intent(DOCUMENT_QUERY, "fallback", reason)
    return _intent(GENERAL, "fallback", reason)
