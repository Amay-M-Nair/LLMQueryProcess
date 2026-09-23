"""Collections, and the one property that makes the page safe to host.

A collection is a namespace. On one machine every visitor is the same person
and a shared index is correct; on a URL it is a document leak, so the page
gives each browser session its own. Two things must hold or that promise is
worthless: no collection may retrieve another's documents, and no name may
become a path outside the collections directory.

These tests drive api/collections.py directly. They used to run over HTTP
against the FastAPI service, which has been removed - the guarantees being
checked were never the routes', they were always this module's.
"""

import pytest

from api import collections
from api.collections import BadCollectionName

SECRET = "BLUEJAY-7741"
PRIVATE = f"The vault access code is {SECRET} and the keyholder is Okonkwo."


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    """Never touch the real data directory."""
    monkeypatch.setattr(collections, "ROOT", tmp_path / "collections")


def upload(name, filename, text):
    """Save and index one document, as the page does for an upload."""
    paths = collections.save_uploads(name, [(filename, text.encode())])
    return collections.ingest(name, paths)


# --- Isolation -------------------------------------------------------------

def test_one_collection_cannot_see_another():
    upload("alice", "private.txt", PRIVATE)

    mine = collections.describe("alice")
    theirs = collections.describe("bob")

    assert mine["chunks"] > 0
    assert theirs["chunks"] == 0
    assert theirs["documents"] == []


def test_a_stranger_retrieves_nothing():
    upload("alice", "private.txt", PRIVATE)
    assert collections.load("bob") is None, "a collection with no uploads has no index"


def test_the_secret_is_only_in_the_owners_index():
    upload("alice", "private.txt", PRIVATE)

    owner = collections.load("alice")
    assert any(SECRET in chunk.text for chunk in owner.chunks)
    assert collections.load("bob") is None


def test_clearing_one_leaves_the_other():
    upload("alice", "a.txt", "Alice's notes about the loading bay rota.")
    upload("bob", "b.txt", "Bob's notes about the delivery schedule.")

    assert collections.clear("alice") is True

    assert collections.describe("alice")["chunks"] == 0
    assert collections.describe("bob")["chunks"] > 0


def test_clearing_removes_the_uploaded_file_too():
    upload("alice", "private.txt", PRIVATE)
    uploaded = collections.upload_dir("alice") / "private.txt"
    assert uploaded.exists()

    collections.clear("alice")
    assert not uploaded.exists(), "the document was left on the server"


# --- Names become directories, so they are checked hard --------------------

@pytest.mark.parametrize(
    "name",
    ["../etc", "..", ".", "a/b", "a\b", "", " ", "-leading", "x" * 65,
     "with space", "semi;colon", "null\x00byte"],
)
def test_a_dangerous_name_is_refused(name):
    with pytest.raises(BadCollectionName):
        collections.normalise(name)


@pytest.mark.parametrize("name", ["alice", "Alice", "team-1", "a_b_c", "x", "9lives"])
def test_a_reasonable_name_is_accepted(name):
    assert collections.normalise(name) == name.lower()


def test_names_differing_only_in_case_are_the_same_collection():
    """Two directories differing by case would collide on Windows anyway."""
    assert collections.normalise("Alice") == collections.normalise("alice")


def test_an_index_stays_under_the_collections_root(tmp_path):
    directory = collections.index_dir("alice")
    assert collections.ROOT.resolve() in directory.resolve().parents


def test_an_uploaded_filename_cannot_escape():
    """The name that arrives is not the name on disk."""
    collections.save_uploads("alice", [("../../escaped.txt", b"content here")])
    written = list(collections.upload_dir("alice").iterdir())
    assert [p.name for p in written] == ["escaped.txt"]


def test_a_hidden_filename_is_refused():
    with pytest.raises(BadCollectionName):
        collections.save_uploads("alice", [(".env", b"GOOGLE_API_KEY=secret")])


# --- Indexing behaviour ----------------------------------------------------

def test_describe_lists_what_was_indexed():
    upload("alice", "rota.txt", "The loading bay rota for the week ahead.")

    body = collections.describe("alice")
    assert [d["name"] for d in body["documents"]] == ["rota.txt"]
    assert body["chunks"] >= 1


def test_the_same_file_twice_is_not_indexed_twice():
    text = "Refunds are available within 30 days of purchase."
    _, first = upload("alice", "refunds.txt", text)
    _, second = upload("alice", "refunds.txt", text)

    assert first.indexed == ["refunds.txt"]
    assert second.indexed == []
    assert second.skipped[0][1] == "already indexed"
    assert second.total_chunks == first.total_chunks


def test_an_edited_file_replaces_its_chunks():
    upload("alice", "rota.txt", "The rota says Tuesday is the delivery day.")
    _, report = upload("alice", "rota.txt",
                       "The rota says Thursday is the delivery day now.")

    assert report.reindexed == ["rota.txt"]
    store = collections.load("alice")
    assert not any("Tuesday" in c.text for c in store.chunks), "stale text survived"
