"""Setup checker.  Run it with:

    .venv/Scripts/python.exe -m utils.check
    .venv/Scripts/python.exe -m utils.check --models
    .venv/Scripts/python.exe -m utils.check --live

`--live` is the one that tells you whether the thing actually works. Without
it this only reports whether a key is present, which is a different question:
a key can be present and expired, rejected, out of quota, or pointed at a
model that has been retired. Those all look READY here and fail on the first
real question.

It spends two API calls: one classification, one short generation.
"""

import sys
import time

from backend.llm import PROVIDERS, ProviderError, ProviderNotReady, get_provider, model_for
from utils import config, prompts


def report_readiness() -> list[str]:
    """Print which providers are configured. Returns the ready ones."""
    print(f"Configured provider: {config.PROVIDER}\n")
    ready = []

    for name in PROVIDERS:
        provider = get_provider(name)
        marker = "->" if name == config.PROVIDER else "  "
        try:
            provider.check_ready()
            print(f"{marker} {name:10} READY      {model_for(name)}")
            ready.append(name)
        except ProviderNotReady as exc:
            first_line = str(exc).splitlines()[0]
            print(f"{marker} {name:10} not ready  {first_line}")

    return ready


def live_check(name: str) -> bool:
    """Make real calls and report what came back. True if both worked."""
    provider = get_provider(name)
    print(f"\nLive check: {name} ({model_for(name)})")
    print("-" * 60)

    # 1. Classification. This is the call most likely to disappoint: it asks
    #    for JSON and nothing else, and a model that answers the question
    #    instead of classifying it breaks routing in a way no amount of
    #    prompt-side wishing fixes.
    question = "What is our refund policy?"
    system, user = prompts.intent_prompt(question, ["employee_handbook.pdf"])
    print(f"  classify {question!r}")
    try:
        started = time.monotonic()
        reply = provider.complete(system, user)
        elapsed = time.monotonic() - started
    except (ProviderNotReady, ProviderError) as exc:
        print(f"    FAILED  {exc}")
        return False

    print(f"    replied in {elapsed:.1f}s: {reply.strip()[:120]!r}")

    from backend.intent_classifier import parse_reply

    parsed = parse_reply(reply)
    if parsed is None:
        print("    UNUSABLE - could not read an intent out of that.")
        print("    Routing still works (it falls back), but every ambiguous")
        print("    question will pay for this call and learn nothing.")
        classification_ok = False
    else:
        print(f"    parsed as: intent={parsed[0]} requires_retrieval={parsed[1]}")
        classification_ok = True

    # 2. Generation, with the citation rule the RAG path depends on.
    print("\n  generate a cited answer from one excerpt")
    excerpt = (
        "[1] Source: refund_policy.pdf p.1\n"
        "Customers may request a full refund within 30 days of purchase."
    )
    try:
        started = time.monotonic()
        answer = provider.complete(
            prompts.RAG_SYSTEM,
            f"Here are the source excerpts:\n\n{excerpt}\n\n"
            "---\n\nQuestion: How long do I have to request a refund?",
        )
        elapsed = time.monotonic() - started
    except (ProviderNotReady, ProviderError) as exc:
        print(f"    FAILED  {exc}")
        return False

    print(f"    replied in {elapsed:.1f}s: {answer.strip()[:200]!r}")

    cited = "[1]" in answer
    correct = "30" in answer
    print(f"    mentions 30 days: {'yes' if correct else 'NO'}")
    print(f"    carries a [1] citation: {'yes' if cited else 'NO'}")
    if not cited:
        print("    (the model is ignoring the citation rule - expect answers")
        print("     whose sources cannot be checked)")

    return classification_ok and cited and correct


def main(argv: list[str]) -> int:
    ready = report_readiness()

    if "--models" in argv:
        print("\nGemini models available to your key:")
        try:
            for name in get_provider("gemini").list_models():
                print(f"  {name}")
        except Exception as exc:
            print(f"  could not list models: {exc}")

    if "--live" in argv:
        if not ready:
            print("\nNothing to call. Set up a provider first - see the README.")
            return 1

        target = config.PROVIDER if config.PROVIDER in ready else ready[0]
        if target != config.PROVIDER:
            print(f"\n{config.PROVIDER} is not ready; testing {target} instead.")

        if not live_check(target):
            print("\nThe provider answered, but not usably. See above.")
            return 1
        print("\nWorks end to end.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
