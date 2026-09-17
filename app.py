"""The web page. Run it with:  streamlit run app.py"""

import os

import streamlit as st

from backend import rag
from backend.llm import (
    PROVIDERS,
    ProviderError,
    ProviderNotReady,
    get_provider,
    model_for,
)
from ingestion.pipeline import build_index, retrieve
from utils import config
from vectorstore.faiss_store import INDEX_FILE, VectorStore
from vectorstore.metadata_store import METADATA_FILE

st.set_page_config(page_title="Document Q&A", layout="wide")


def render_sources(sources: list[tuple[str, float, str]]) -> None:
    """Show the excerpts the answer was built from, numbered to match its citations."""
    with st.expander(f"Sources ({len(sources)})"):
        for number, (label, score, text) in enumerate(sources, start=1):
            st.markdown(f"**[{number}] {label}** - similarity {score:.3f}")
            st.caption(text)


# --- State -----------------------------------------------------------------
if "store" not in st.session_state:
    try:
        st.session_state.store = VectorStore.load(config.INDEX_DIR)
    except Exception as exc:
        # An index left by an older version, or a half-written one. Say so
        # rather than crashing on load with a stack trace.
        st.session_state.store = None
        st.session_state.load_error = str(exc)
if "history" not in st.session_state:
    st.session_state.history = []
if "provider_name" not in st.session_state:
    st.session_state.provider_name = config.PROVIDER
if "api_keys" not in st.session_state:
    # Keys typed into the sidebar, per provider. Session state is held server
    # side and per browser session, so one visitor's key is never handed to
    # another - but it is also never persisted, by design.
    st.session_state.api_keys = {}

store: VectorStore | None = st.session_state.store

if st.session_state.get("load_error"):
    st.error(f"Could not open the saved index: {st.session_state.load_error}")


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Documents")

    uploads = st.file_uploader(
        "Add PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Add to index", type="primary", disabled=not uploads):
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for upload in uploads:
            destination = config.UPLOAD_DIR / upload.name
            destination.write_bytes(upload.getbuffer())
            saved_paths.append(destination)

        status = st.status("Indexing...", expanded=True)
        try:
            new_store, report = build_index(
                saved_paths,
                existing=store,
                on_progress=lambda message: status.write(message),
            )
            new_store.save(config.INDEX_DIR)
            st.session_state.store = new_store
            store = new_store

            status.update(label=f"Added {report.chunks_added} chunks", state="complete")
            for name in report.reindexed:
                st.info(f"Re-indexed **{name}** - its contents had changed")
            for name, reason in report.skipped:
                st.warning(f"Skipped **{name}** - {reason}")
            if report.changed:
                st.rerun()
        except Exception as exc:
            status.update(label="Indexing failed", state="error")
            st.exception(exc)

    st.divider()

    if store and len(store) > 0:
        st.metric("Chunks indexed", len(store))
        st.caption("Files in the index:")
        for name, record in store.documents.items():
            count = record.chunk_count
            st.caption(f"- {name} ({count} chunk{'' if count == 1 else 's'})")

        if st.button("Clear index"):
            for filename in (INDEX_FILE, METADATA_FILE):
                (config.INDEX_DIR / filename).unlink(missing_ok=True)
            st.session_state.store = None
            st.session_state.history = []
            st.session_state.load_error = None
            st.rerun()
    elif st.session_state.get("load_error"):
        # The index exists but could not be opened, so the usual Clear button
        # above is out of reach - offer it here or there is no way out.
        if st.button("Delete the unreadable index"):
            for filename in (INDEX_FILE, METADATA_FILE):
                (config.INDEX_DIR / filename).unlink(missing_ok=True)
            st.session_state.load_error = None
            st.rerun()
    else:
        st.info("No documents indexed yet. Upload a file above to start.")

    st.divider()
    st.header("Answer model")

    provider_names = list(PROVIDERS)
    st.session_state.provider_name = st.selectbox(
        "Provider",
        provider_names,
        index=provider_names.index(st.session_state.provider_name),
        help="Retrieval always runs locally and is free. Only this step calls out.",
    )

    provider_name = st.session_state.provider_name
    keys = st.session_state.api_keys

    # The key box exists so a deployed copy can ask each visitor for their own
    # key rather than shipping one. Typed keys live in this browser session
    # only: never written to disk, never logged, gone when the tab closes.
    if get_provider(provider_name).needs_key:
        variable = get_provider(provider_name).key_variable
        from_env = bool(os.environ.get(variable))

        typed = st.text_input(
            "API key",
            value=keys.get(provider_name, ""),
            type="password",
            placeholder="Using the key from .env" if from_env else f"Paste your {variable}",
            help=(
                "Overrides .env for this session only. Nothing is saved to "
                "disk, so closing the tab forgets it."
            ),
        )
        keys[provider_name] = typed

        if typed:
            st.caption("Using the key you entered (this session only).")
        elif from_env:
            st.caption(f"Using `{variable}` from your .env file.")

        # "Ready" below only means a key is present. A key can be present and
        # be a typo, expired, or out of quota - all of which look identical
        # until the first question fails. This spends one request to find out,
        # on a button so it is never spent behind your back.
        if st.button("Test this key", help="Makes one real API call."):
            candidate = get_provider(provider_name, api_key=typed or None)
            with st.spinner("Calling the API..."):
                try:
                    reply = candidate.complete("Reply with the single word OK.", "Ready?")
                    st.success(f"The key works. Replied: {reply.strip()[:40]!r}")
                except Exception as exc:
                    st.error(f"That key did not work: {exc}")

    provider = get_provider(provider_name, api_key=keys.get(provider_name) or None)
    st.caption(f"Model: `{model_for(provider.name)}`")
    st.caption(provider.note)

    try:
        provider.check_ready()
        st.success("Ready", icon=":material/check_circle:")
        provider_ready = True
    except ProviderNotReady as exc:
        st.warning(str(exc))
        provider_ready = False


# --- Main ------------------------------------------------------------------
st.title("Ask your documents")
st.caption(
    "Answers come only from the files you uploaded. Every claim is cited back "
    "to the page it came from."
)

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        render_sources(entry["sources"])

has_docs = bool(store and len(store) > 0)

if not has_docs:
    placeholder = "Upload a document first"
elif not provider_ready:
    placeholder = f"Set up {provider.name} first - see the sidebar"
else:
    placeholder = "Ask a question about your documents"

question = st.chat_input(placeholder, disabled=not (has_docs and provider_ready))

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            retrieved = retrieve(store, question, k=config.TOP_K)

        if not retrieved:
            st.warning(
                "Nothing in your documents was close enough to that question. "
                "Try rephrasing, or lower MIN_SIMILARITY in utils/config.py."
            )
        else:
            answer = None
            try:
                answer = st.write_stream(
                    rag.stream_answer(question, retrieved, provider=provider)
                )
            except (ProviderNotReady, ProviderError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The request failed: {exc}")

            sources = [(chunk.label, score, chunk.text) for chunk, score in retrieved]
            render_sources(sources)

            if answer:
                st.session_state.history.append(
                    {"question": question, "answer": answer, "sources": sources}
                )
