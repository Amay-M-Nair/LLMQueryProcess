"""Setup checker.  Run it with:

    .venv/Scripts/python.exe -m utils.check
    .venv/Scripts/python.exe -m utils.check --models
"""

import sys

from backend.llm import PROVIDERS, ProviderNotReady, get_provider, model_for
from utils import config


def main(argv: list[str]) -> int:
    print(f"Configured provider: {config.PROVIDER}\n")

    for name in PROVIDERS:
        provider = get_provider(name)
        marker = "->" if name == config.PROVIDER else "  "
        try:
            provider.check_ready()
            print(f"{marker} {name:10} READY      {model_for(name)}")
        except ProviderNotReady as exc:
            first_line = str(exc).splitlines()[0]
            print(f"{marker} {name:10} not ready  {first_line}")

    if "--models" in argv:
        print("\nGemini models available to your key:")
        try:
            for name in get_provider("gemini").list_models():
                print(f"  {name}")
        except Exception as exc:
            print(f"  could not list models: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
