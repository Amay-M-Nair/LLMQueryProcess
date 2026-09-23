"""The API, and the one property that makes it safe to expose.

A collection is a name the caller chooses. Two things must hold or the
service cannot be put behind a URL: no collection may retrieve another's
documents, and no name may become a path outside the collections directory.

The answering routes are exercised without a model. What happens to a
question is backend/query_processor.py's business and is tested there; what
matters here is that the route resolves the right index and streams back what
it was given.
"""

import shutil

import pytest
from fastapi.testclient import TestClient

from api import collections
from api.collections import BadCollectionName
from api.main import app

SECRET = "BLUEJAY-7741"
PRIVATE = f"The vault access code is {SECRET} and the keyholder is Okonkwo."


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(collections, "ROOT", tmp_path / "collections")
    return TestClient(app)


def upload(client, collection, name, text):
    return client.post(
        f"/collections/{collection}/documents",
        files=[("files", (name, text.encode(), "text/plain"))],
    )


# --- Isolation -------------------------------------------------------------

def test_one_collection_cannot_see_another(client):
    assert upload(client, "alice", "private.txt", PRIVATE).status_code == 200

    mine = client.get("/collections/alice").json()
    theirs = client.get("/collections/bob").json()

    assert mine["chunks"] > 0
    assert theirs["chunks"] == 0
    assert theirs["documents"] == []


def test_a_stranger_retrieves_nothing(client, monkeypatch):
    upload(client, "alice", "private.txt", PRIVATE)

    store = collections.load("bob")
    assert store is None, "a collection with no uploads should have no index"


def test_the_secret_is_only_in_the_owners_index(client):
    upload(client, "alice", "private.txt", PRIVATE)

    owner = collections.load("alice")
    assert any(SECRET in chunk.text for chunk in owner.chunks)
    assert collections.load("bob") is None


def test_clearing_one_leaves_the_other(client):
    upload(client, "alice", "a.txt", "Alice's notes about the loading bay rota.")
    upload(client, "bob", "b.txt", "Bob's notes about the delivery schedule.")

    assert client.delete("/collections/alice").json()["existed"] is True

    assert client.get("/collections/alice").json()["chunks"] == 0
    assert client.get("/collections/bob").json()["chunks"] > 0


def test_clearing_removes_the_uploaded_file_too(client, tmp_path):
    upload(client, "alice", "private.txt", PRIVATE)
    uploaded = collections.upload_dir("alice") / "private.txt"
    assert uploaded.exists()

    client.delete("/collections/alice")
    assert not uploaded.exists(), "the document was left on the server"


# --- Names become directories, so they are checked hard --------------------

@pytest.mark.parametrize(
    "name",
    ["../etc", "..", ".", "a/b", "a\\b", "", " ", "-leading", "x" * 65,
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


def test_traversal_is_refused_over_http(client):
    response = client.get("/collections/..%2F..%2Fetc")
    assert response.status_code in (404, 422), response.status_code


def test_an_index_stays_under_the_collections_root(tmp_path, monkeypatch):
    monkeypatch.setattr(collections, "ROOT", tmp_path / "collections")
    directory = collections.index_dir("alice")
    assert (tmp_path / "collections").resolve() in directory.resolve().parents


def test_an_uploaded_filename_cannot_escape(client):
    """The name in the request is not the name on disk."""
    response = client.post(
        "/collections/alice/documents",
        files=[("files", ("../../escaped.txt", b"content here", "text/plain"))],
    )
    assert response.status_code == 200
    written = list(collections.upload_dir("alice").iterdir())
    assert [p.name for p in written] == ["escaped.txt"]


# --- The ordinary routes ---------------------------------------------------

def test_health_reports_the_model(client):
    body = client.get("/health").json()
    assert body["provider"] in body["providers"]
    assert "lengths" in body and body["lengths"]


def test_describe_lists_what_was_indexed(client):
    upload(client, "alice", "rota.txt", "The loading bay rota for the week ahead.")
    body = client.get("/collections/alice").json()
    assert [d["name"] for d in body["documents"]] == ["rota.txt"]
    assert body["chunks"] >= 1


def test_the_same_file_twice_is_not_indexed_twice(client):
    text = "Refunds are available within 30 days of purchase."
    first = upload(client, "alice", "refunds.txt", text).json()
    second = upload(client, "alice", "refunds.txt", text).json()

    assert first["indexed"] == ["refunds.txt"]
    assert second["indexed"] == []
    assert second["skipped"][0]["reason"] == "already indexed"
    assert second["chunks"] == first["chunks"]


def test_an_edited_file_replaces_its_chunks(client):
    upload(client, "alice", "rota.txt", "The rota says Tuesday is the delivery day.")
    body = upload(client, "alice", "rota.txt",
                  "The rota says Thursday is the delivery day now.").json()

    assert body["reindexed"] == ["rota.txt"]
    store = collections.load("alice")
    assert not any("Tuesday" in c.text for c in store.chunks), "stale text survived"


def test_uploading_nothing_is_a_422(client):
    assert client.post("/collections/alice/documents", files=[]).status_code == 422


def test_bad_history_is_rejected_rather_than_ignored(client):
    response = client.post(
        "/collections/alice/ask",
        data={"question": "hello", "history": "not json"},
    )
    assert response.status_code == 422


def test_the_openapi_document_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Azriel"
    assert "/collections/{collection}/ask" in schema["paths"]


def test_the_root_sends_a_browser_to_the_docs(client):
    """The default answer there is {"detail":"Not Found"}, which helps nobody."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_the_docs_page_renders(client):
    assert client.get("/docs").status_code == 200
