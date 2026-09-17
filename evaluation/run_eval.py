"""Measure whether the query layer earns its keep.

    python -m evaluation.run_eval                 # routing + retrieval, cheap
    python -m evaluation.run_eval --answers       # also generate and score answers
    python -m evaluation.run_eval --provider ollama

Adding components is easy; showing they help is the part that takes evidence.
This scores the three things that could each be quietly wrong:

  intent accuracy    - is the question sent down the right path at all?
  retrieval relevance - do the chunks come from the document that holds the answer?
  answer faithfulness - do the citations resolve, and does the answer avoid
                        claiming documentary support it does not have?

plus what each question costs in calls and seconds.

Retrieval is forced on, even though the app skips it for an index this small
(see FULL_CONTEXT_WORDS). Measuring hit@k against a run that returned every
chunk would score 1.0 and mean nothing.
"""

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.llm import get_provider
from backend.query_processor import Plan, process
from ingestion.pipeline import build_index
from utils import config

def say(message: str = "") -> None:
    """Print and flush.

    A long run is worth watching, and Python buffers stdout when it is piped
    to a file - which turns a ten-minute evaluation into ten minutes of
    silence followed by everything at once.
    """
    print(message, flush=True)


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
DATASET = HERE / "dataset.jsonl"
INDEX = config.PROJECT_ROOT / "data" / "eval_index"

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
UNSUPPORTED_MARKER = "beyond your documents"
# Phrases a model reaches for when the excerpts do not answer the question.
ADMITS_MISS = (
    "do not contain", "does not contain", "not contain", "no information",
    "not mentioned", "not specified", "not found", "do not provide",
    "does not provide", "not covered", "cannot be found", "unable to find",
    "not included", "do not mention", "does not mention",
)


@dataclass
class Result:
    case: dict
    plan: Plan
    answer: str = ""
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.case["id"]


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_eval_index():
    """A fresh index over the evaluation corpus, rebuilt every run."""
    shutil.rmtree(INDEX, ignore_errors=True)
    store, report = build_index(sorted(CORPUS.glob("*.txt")))
    if report.skipped:
        say(f"  warning: skipped {report.skipped}")
    store.save(INDEX)
    return store


def cited_numbers(answer: str) -> set[int]:
    found = set()
    for group in CITATION.findall(answer):
        for part in group.split(","):
            found.add(int(part.strip()))
    return found


def admits_the_miss(answer: str) -> bool:
    lowered = answer.lower()
    return UNSUPPORTED_MARKER in lowered or any(p in lowered for p in ADMITS_MISS)


def score(result: Result, check_answers: bool) -> None:
    """Record every way this case fell short. Empty means it passed."""
    case, plan, trace = result.case, result.plan, result.plan.trace

    expected_intent = case.get("expected_intent")
    if expected_intent and trace.intent.name != expected_intent:
        result.failures.append(f"intent {trace.intent.name} != {expected_intent}")

    expected_route = case.get("expected_route")
    if expected_route and trace.route.name != expected_route:
        result.failures.append(f"route {trace.route.name} != {expected_route}")

    expected_source = case.get("expected_source")
    if expected_source:
        sources = [chunk.source for chunk, _ in plan.sources]
        if expected_source not in sources:
            result.failures.append(
                f"retrieval missed {expected_source} (got {sorted(set(sources)) or 'nothing'})"
            )

    if case.get("expect_rewrite") and not (trace.rewrite and trace.rewrite.changed):
        reason = trace.rewrite.reason if trace.rewrite else "no rewrite attempted"
        result.failures.append(f"expected a rewrite ({reason})")

    if case["expected_intent"] == "calculation" and trace.api_calls:
        result.failures.append(f"arithmetic cost {trace.api_calls} API calls")

    if not check_answers:
        return

    answer = result.answer
    if not answer:
        result.failures.append("no answer produced")
        return

    # "2.0|double" means either will do: a fact can be stated correctly in
    # more than one way, and demanding every phrasing marks right answers
    # wrong.
    for keyword in case.get("expected_keywords", []):
        alternatives = [k.strip().lower() for k in keyword.split("|")]
        if not any(a in answer.lower() for a in alternatives):
            result.failures.append(f"answer omits {keyword!r}")

    # Faithfulness: a citation that points past the excerpts actually supplied
    # is a fabricated source, which is worse than no citation at all.
    numbers = cited_numbers(answer)
    supplied = len(plan.sources)
    dangling = {n for n in numbers if n < 1 or n > supplied}
    if dangling:
        result.failures.append(f"cites excerpts that do not exist: {sorted(dangling)}")

    # Only meaningful when documents were actually consulted. On the direct
    # route nothing was retrieved and nothing was claimed, so there is no
    # documentary support to have overstated.
    if case.get("expect_unsupported") and plan.sources:
        if not admits_the_miss(answer):
            result.failures.append("answered as though documented when it is not")
        if numbers and UNSUPPORTED_MARKER not in answer.lower():
            result.failures.append("cited sources for an unsupported answer")


def run(provider_name: str, check_answers: bool, limit: int | None) -> int:
    say("Building the evaluation index...")
    store = build_eval_index()
    say(f"  {len(store)} chunks from {', '.join(store.sources)}\n")

    provider = get_provider(provider_name)
    provider.check_ready()

    cases = load_cases()
    if limit:
        cases = cases[:limit]

    results: list[Result] = []
    answers_by_id: dict[str, str] = {}

    for case in cases:
        # A follow-up is only a follow-up if the turn before it is present.
        history = []
        if case.get("follows"):
            previous = next(c for c in cases if c["id"] == case["follows"])
            history = [{
                "question": previous["query"],
                "answer": answers_by_id.get(previous["id"], ""),
            }]

        started = time.monotonic()
        plan = process(case["query"], store=store, history=history, provider=provider)
        answer = ""
        if check_answers:
            try:
                if plan.answer is not None:
                    answer = plan.answer
                elif plan.message is not None:
                    answer = plan.message
                elif plan.stream is not None:
                    answer = "".join(plan.stream())
            except Exception as exc:
                answer = ""
                say(f"  {case['id']}: generation failed - {exc}")
        elapsed = time.monotonic() - started

        result = Result(case, plan, answer, elapsed)
        score(result, check_answers)
        results.append(result)
        answers_by_id[case["id"]] = answer

        mark = "ok  " if not result.failures else "FAIL"
        say(f"  [{mark}] {case['id']:11} {case['query'][:52]:54} "
              f"{plan.trace.route.name:9} {plan.trace.api_calls} call(s) {elapsed:5.1f}s")
        for failure in result.failures:
            say(f"         - {failure}")

    report(results, check_answers, len(store))
    shutil.rmtree(INDEX, ignore_errors=True)
    return 1 if any(r.failures for r in results) else 0


def report(results: list[Result], check_answers: bool, index_size: int) -> None:
    say("\n" + "=" * 74)
    say("RESULTS")
    say("=" * 74)

    # --- Intent accuracy, split by how it was decided ---------------------
    labelled = [r for r in results if r.case.get("expected_intent")]
    correct = [r for r in labelled if r.plan.trace.intent.name == r.case["expected_intent"]]
    say(f"\nIntent accuracy      {len(correct)}/{len(labelled)} "
          f"({100 * len(correct) / max(len(labelled), 1):.0f}%)")

    by_method: dict[str, list[bool]] = {}
    for r in labelled:
        hit = r.plan.trace.intent.name == r.case["expected_intent"]
        by_method.setdefault(r.plan.trace.intent.method, []).append(hit)
    for method, hits in sorted(by_method.items()):
        say(f"  decided by {method:10} {sum(hits)}/{len(hits)} correct")

    wrong = [r for r in labelled if r.plan.trace.intent.name != r.case["expected_intent"]]
    for r in wrong:
        say(f"  MISSED {r.id}: {r.case['expected_intent']} -> "
              f"{r.plan.trace.intent.name} ({r.plan.trace.intent.method})")

    # --- Retrieval --------------------------------------------------------
    graded = [r for r in results if r.case.get("expected_source")]
    hits, reciprocal = 0, 0.0
    for r in graded:
        sources = [chunk.source for chunk, _ in r.plan.sources]
        wanted = r.case["expected_source"]
        if wanted in sources:
            hits += 1
            reciprocal += 1 / (sources.index(wanted) + 1)
    total = max(len(graded), 1)
    say(f"\nRetrieval hit@{config.TOP_K}       {hits}/{len(graded)} "
          f"({100 * hits / total:.0f}%)")
    say(f"Retrieval MRR        {reciprocal / total:.3f}")

    # A score is only worth as much as the test behind it. Returning most of
    # the index cannot fail, so state what share was returned rather than let
    # a meaningless 100% pass for a good one.
    returned = min(config.TOP_K, index_size)
    share = returned / max(index_size, 1)
    say(f"  retrieved {returned} of {index_size} chunks "
        f"({100 * share:.0f}% of the index) per question")
    if share > 0.25:
        say("  WARNING: that is a large share of the corpus, so hit@k can")
        say("  barely fail. Treat the score above as weak evidence - grow the")
        say("  corpus or lower --top-k to make it mean something.")

    # --- Answers ----------------------------------------------------------
    if check_answers:
        answered = [r for r in results if r.answer]
        keyword_ok = [
            r for r in answered
            if r.case.get("expected_keywords")
            and not any(f.startswith("answer omits") for f in r.failures)
        ]
        expected_keywords = [r for r in answered if r.case.get("expected_keywords")]
        say(f"\nAnswer contains the expected fact  {len(keyword_ok)}/"
              f"{len(expected_keywords)}")

        fabricated = [r for r in answered
                      if any("do not exist" in f for f in r.failures)]
        say(f"Citations that resolve             "
              f"{len(answered) - len(fabricated)}/{len(answered)}")

        unsupported = [r for r in answered if r.case.get("expect_unsupported")]
        honest = [r for r in unsupported
                  if not any("as though documented" in f for f in r.failures)]
        if unsupported:
            say(f"Absent facts admitted as absent    {len(honest)}/{len(unsupported)}")

    # --- Cost -------------------------------------------------------------
    calls = sum(r.plan.trace.api_calls for r in results)
    free = [r for r in results if r.plan.trace.api_calls == 0]
    seconds = [r.seconds for r in results]
    say(f"\nAPI calls            {calls} total, "
          f"{calls / max(len(results), 1):.2f} per question")
    say(f"Answered for free    {len(free)}/{len(results)} questions")
    say(f"Latency              median {sorted(seconds)[len(seconds) // 2]:.2f}s, "
          f"worst {max(seconds):.2f}s")

    stages: dict[str, list[float]] = {}
    for r in results:
        for stage, value in r.plan.trace.timings.items():
            stages.setdefault(stage, []).append(value)
    for stage, values in sorted(stages.items()):
        say(f"  {stage:10} {sum(values) / len(values) * 1000:7.0f}ms average "
              f"over {len(values)} questions")

    failed = [r for r in results if r.failures]
    say(f"\n{len(results) - len(failed)}/{len(results)} cases clean.")
    if failed:
        say("Fell short: " + ", ".join(r.id for r in failed))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=config.PROVIDER)
    parser.add_argument("--answers", action="store_true",
                        help="also generate answers and score them (costs a call each)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None,
                        help="override TOP_K; lower makes retrieval harder to pass")
    args = parser.parse_args(argv)

    # Retrieval must actually run for hit@k to mean anything.
    config.FULL_CONTEXT_WORDS = 0
    if args.top_k:
        config.TOP_K = args.top_k
    return run(args.provider, args.answers, args.limit)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
