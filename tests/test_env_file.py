"""Writing a key into .env must not damage the rest of the file.

.env is hand-edited and full of comments explaining where each key comes
from. A writer that reformats it, or drops a commented-out line, loses work
that is not its to lose.
"""

import pytest

from utils import env_file
from utils.env_file import EnvWriteError

SAMPLE = """# Copy this file to `.env` and fill in the key for whichever provider you use.
# Which one is used is set by PROVIDER in utils/config.py (default: gemini).

# Gemini - free tier, no card required.  Get one at:
#   https://aistudio.google.com/apikey
GOOGLE_API_KEY=old-value

# Claude - paid, needs credit on the account.
# ANTHROPIC_API_KEY=

# Ollama needs no key.
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(SAMPLE, encoding="utf-8")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return path


def test_replaces_an_existing_value(env):
    replaced = env_file.set_value("GOOGLE_API_KEY", "new-value", env)
    assert replaced is True
    assert "GOOGLE_API_KEY=new-value" in env.read_text(encoding="utf-8")
    assert "old-value" not in env.read_text(encoding="utf-8")


def test_keeps_every_comment_and_blank_line(env):
    before = env.read_text(encoding="utf-8").splitlines()
    env_file.set_value("GOOGLE_API_KEY", "new-value", env)
    after = env.read_text(encoding="utf-8").splitlines()

    comments_before = [l for l in before if l.startswith("#") or not l.strip()]
    comments_after = [l for l in after if l.startswith("#") or not l.strip()]
    assert comments_before == comments_after
    assert len(before) == len(after), "a line was added or lost"


def test_a_commented_out_variable_is_not_mistaken_for_the_real_one(env):
    """`# ANTHROPIC_API_KEY=` must stay commented, and a real one be added."""
    env_file.set_value("ANTHROPIC_API_KEY", "sk-test", env)
    text = env.read_text(encoding="utf-8")
    assert "# ANTHROPIC_API_KEY=" in text, "the commented line was overwritten"
    assert "\nANTHROPIC_API_KEY=sk-test" in text


def test_appends_when_absent(env):
    added = env_file.set_value("BRAND_NEW_KEY", "value", env)
    assert added is False
    assert env.read_text(encoding="utf-8").rstrip().endswith("BRAND_NEW_KEY=value")


def test_creates_the_file_when_missing(tmp_path):
    path = tmp_path / ".env"
    env_file.set_value("GOOGLE_API_KEY", "value", path)
    assert path.read_text(encoding="utf-8").strip() == "GOOGLE_API_KEY=value"


def test_handles_an_export_prefix(env):
    env.write_text("export GOOGLE_API_KEY=old\n", encoding="utf-8")
    env_file.set_value("GOOGLE_API_KEY", "new", env)
    assert env.read_text(encoding="utf-8").strip() == "GOOGLE_API_KEY=new"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("simple-key", "simple-key"),
        ("has space", '"has space"'),
        ("has#hash", '"has#hash"'),
        ("", '""'),
    ],
)
def test_quotes_only_when_needed(value, expected):
    assert env_file.quote(value) == expected


def test_refuses_a_value_containing_a_newline(env):
    """A line break would silently truncate the key and corrupt the file."""
    with pytest.raises(EnvWriteError):
        env_file.set_value("GOOGLE_API_KEY", "abc\ndef", env)
    assert "old-value" in env.read_text(encoding="utf-8"), "the file was modified"


def test_refuses_a_bad_variable_name(env):
    with pytest.raises(EnvWriteError):
        env_file.set_value("not a name", "value", env)


def test_the_running_process_sees_the_new_value(env, monkeypatch):
    import os
    env_file.set_value("GOOGLE_API_KEY", "live-value", env)
    assert os.environ["GOOGLE_API_KEY"] == "live-value"


def test_clearing_removes_the_line_and_the_variable(env):
    import os
    env_file.set_value("GOOGLE_API_KEY", "value", env)
    assert env_file.clear_value("GOOGLE_API_KEY", env) is True
    text = env.read_text(encoding="utf-8")
    assert "GOOGLE_API_KEY=" not in text.replace("# ANTHROPIC_API_KEY=", "")
    assert "GOOGLE_API_KEY" not in os.environ
    assert "# Gemini - free tier" in text, "clearing took the comments with it"


def test_clearing_something_absent_is_not_an_error(env):
    assert env_file.clear_value("NEVER_SET", env) is False
