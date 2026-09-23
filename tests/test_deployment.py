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


def test_the_dockerfile_defaults_to_the_safe_settings():
    """An image run without env vars must not come up sharing one index."""
    text = (config.PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AZRIEL_COLLECTIONS=visitor" in text
    assert "AZRIEL_PUBLIC=1" in text


def test_the_dockerfile_bakes_the_embedding_model_in():
    """Otherwise the first question after each deploy waits on a download."""
    text = (config.PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "HuggingFaceEmbeddings" in text


def test_the_dockerignore_keeps_secrets_and_bulk_out():
    text = (config.PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in (".env", ".venv/", "data/", ".git/"):
        assert entry in text, f"{entry} would be copied into the image"
