"""Where the technical detail goes when the user gets a sentence.

Every error in this project is reported twice: once to whoever is using it,
in language that says what to do, and once here, with whatever the exception
actually said. Without the second half a deployed copy loses the detail
entirely - the message on screen is all there is, and it was written to be
reassuring rather than diagnostic.

It matters most where an error is deliberately swallowed. A failed rewrite
falls back to the original question and a failed OCR page is skipped, both
on purpose, and both would otherwise vanish without trace.

Two things are never logged: the contents of anyone's documents, and API
keys. The first because a log file is not where someone's private files
belong, the second because logs get pasted into issues.
"""

import logging
import os
import re
import sys

from utils import config

FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Variables whose values must never be printed, wherever they turn up.
SECRET_VARIABLES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")

# A key in a URL, which is how clients pass them and therefore how they end up
# inside the exceptions those clients raise.
KEY_IN_URL = re.compile(r"([?&](?:key|api_key|apikey|access_token)=)[^&\s\"']+",
                        re.IGNORECASE)
REDACTED = "<redacted>"


def redact(text: str) -> str:
    """Strip anything secret from a message before it is shown or stored.

    Two passes, because neither catches the other's case. Known values cover
    a key that arrives whole, whatever surrounds it. The URL pattern covers a
    key this process has never seen - a second account's, or one typed into
    the sidebar of another session.

    Used on anything derived from an exception, because a client that puts the
    key in the request URL puts it in the error too, and that error reaches
    both the log and the debug panel.
    """
    if not text:
        return text

    cleaned = str(text)
    for variable in SECRET_VARIABLES:
        value = os.environ.get(variable)
        if value and len(value) > 8:
            cleaned = cleaned.replace(value, REDACTED)
    return KEY_IN_URL.sub(lambda m: m.group(1) + REDACTED, cleaned)


class _Redact(logging.Filter):
    """Last line of defence: nothing reaches a handler unredacted.

    Redacting at each call site would work until somebody forgot, and the
    thing they forgot would be a key in a log file.
    """

    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(
                redact(a) if isinstance(a, str) else a for a in record.args
            )
        return True


_configured = False


def setup() -> None:
    """Attach handlers once. Safe to call from anywhere, including a rerun.

    Streamlit re-executes the whole script on every interaction, so this has
    to be idempotent or a log line would appear once more after each click.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger("azriel")
    root.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    root.propagate = False

    root.addFilter(_Redact())

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    root.addHandler(console)

    if config.LOG_FILE:
        try:
            config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            to_file = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
            to_file.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
            root.addHandler(to_file)
        except OSError:
            # A read-only or full disk is not a reason to refuse to run.
            root.warning("could not open %s for writing", config.LOG_FILE)

    _configured = True


def get(name: str) -> logging.Logger:
    """A logger for one module. Use the module's own name."""
    setup()
    return logging.getLogger(f"azriel.{name.rsplit('.', 1)[-1]}")
