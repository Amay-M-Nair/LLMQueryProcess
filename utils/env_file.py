"""Reading and writing single values in the .env file.

A key typed into the sidebar lives in the browser session and is forgotten
when the tab closes, which is right for a shared deployment and tedious on
your own machine. This is what lets the app offer to keep it.

Everything here rewrites one line and leaves the rest of the file exactly as
it was: .env is hand-edited and commented, and a writer that reformats it
would lose work that is not its to lose.

The value is never logged, never echoed, and never returned by anything that
prints. `.env` is in .gitignore, which is what makes writing there safe.
"""

import os
import re
from pathlib import Path

from utils import config

ENV_PATH = config.PROJECT_ROOT / ".env"

# NAME=value, allowing surrounding space and an optional `export` prefix,
# which people copy in from shell instructions.
LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


class EnvWriteError(RuntimeError):
    """The file could not be written. The message is shown to the user."""


def is_valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))


def quote(value: str) -> str:
    """Quote only when the value needs it, so simple keys stay readable."""
    if value == "" or re.search(r"[\s#'\"]", value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def set_value(name: str, value: str, path: Path | None = None) -> bool:
    """Set NAME in .env. Returns True if a line was replaced, False if added.

    The whole file is rewritten in one call rather than appended to, so a
    failure part-way cannot leave a half-written line that breaks parsing on
    the next start.
    """
    if not is_valid_name(name):
        raise EnvWriteError(f"{name!r} is not a usable variable name.")
    if "\n" in value or "\r" in value:
        raise EnvWriteError("A key cannot contain a line break.")

    path = Path(path or ENV_PATH)
    try:
        original = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise EnvWriteError(f"Could not read {path.name}: {exc}") from exc

    lines = original.splitlines()
    replacement = f"{name}={quote(value)}"
    replaced = False

    for index, line in enumerate(lines):
        match = LINE.match(line)
        if match and match.group(1) == name:
            lines[index] = replacement
            replaced = True
            break

    if not replaced:
        # Keep a commented-out placeholder company rather than landing above it.
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(replacement)

    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        raise EnvWriteError(f"Could not write {path.name}: {exc}") from exc

    # So the running process sees it too, without a restart.
    os.environ[name] = value
    return replaced


def clear_value(name: str, path: Path | None = None) -> bool:
    """Remove NAME from .env and from this process. True if it was there."""
    path = Path(path or ENV_PATH)
    if not path.exists():
        os.environ.pop(name, None)
        return False

    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [
        line for line in lines
        if not (LINE.match(line) and LINE.match(line).group(1) == name)
    ]
    removed = len(kept) != len(lines)
    if removed:
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.environ.pop(name, None)
    return removed
