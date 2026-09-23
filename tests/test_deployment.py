"""The settings that decide whether this is safe to expose.

The page defaults to one shared collection, which is right on one machine and
a document leak on a URL. These check that the switch exists, that it does
what it says, and that the two ways of getting it wrong are caught.
"""

import importlib
import os

import pytest

from utils import config


@pytest.fixture
def reloaded(monkeypatch):
    """Re-import config with the environment a deployment would have."""
    def load(**env):
        for key in ("AZRIEL_COLLECTIONS", "AZRIEL_PUBLIC"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)
    yield load
    for key in ("AZRIEL_COLLECTIONS", "AZRIEL_PUBLIC"):
        os.environ.pop(key, None)
    importlib.reload(config)


def test_shared_is_the_default(reloaded):
    """One machine, one user: documents should still be there tomorrow."""
    assert reloaded().COLLECTION_MODE == "shared"
    assert reloaded().PUBLIC is False


def test_visitor_mode_is_opt_in(reloaded):
    assert reloaded(AZRIEL_COLLECTIONS="visitor").COLLECTION_MODE == "visitor"


def test_public_is_opt_in(reloaded):
    assert reloaded(AZRIEL_PUBLIC="1").PUBLIC is True


def test_a_visitor_collection_name_is_valid_and_unguessable():
    """It is the only thing standing between two visitors' documents."""
    import secrets

    from api import collections

    names = {"v" + secrets.token_hex(16) for _ in range(200)}
    assert len(names) == 200, "collection ids collided"
    for name in list(names)[:20]:
        assert collections.normalise(name) == name
        assert len(name) >= 32, "short enough to guess"


def test_the_cpu_torch_pin_survives():
    """Without it the Linux wheel drags in 2.5 GB of CUDA for an absent GPU."""
    text = (config.PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cpu" in text
    lines = [line.strip() for line in text.splitlines()]
    assert any(line.startswith("torch>=") for line in lines), (
        "the index is useless without asking for torch by name"
    )


def test_packages_txt_is_bare_package_names():
    """Community Cloud feeds this to apt-get; a comment would be a package."""
    lines = (config.PROJECT_ROOT / "packages.txt").read_text(
        encoding="utf-8"
    ).split()
    assert lines, "packages.txt is empty"
    for entry in lines:
        assert not entry.startswith("#"), f"{entry!r} would be apt-get installed"
    assert "libgl1" in lines, "OCR imports opencv, which links against libGL"


def test_the_secrets_are_read_from_the_environment(reloaded):
    """Community Cloud sets root-level secrets as environment variables.

    This is the whole deployment contract: paste three lines into Secrets and
    the app comes up isolated. If config stopped reading the environment, a
    deployment would silently come up shared.
    """
    settings = reloaded(AZRIEL_COLLECTIONS="visitor", AZRIEL_PUBLIC="1")
    assert settings.COLLECTION_MODE == "visitor"
    assert settings.PUBLIC is True
