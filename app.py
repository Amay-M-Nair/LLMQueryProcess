"""The web page. Run it with:  streamlit run app.py

Every Streamlit call in the project is in this file, and no decision is. What
happens to a question is decided in backend/query_processor.py; this renders
the result.
"""

import os

import streamlit as st

from backend.llm import PROVIDERS, ProviderError, ProviderNotReady, get_provider, model_for
from backend.query_processor import QueryTrace, process
from ingestion.pipeline import build_index
from utils import config
from vectorstore.faiss_store import INDEX_FILE, VectorStore
from vectorstore.metadata_store import METADATA_FILE

st.set_page_config(page_title="Document Q&A", layout="wide")


def render_sources(sources: list[tuple[str, float, str]]) -> None:
    """The excerpts an answer was built from, numbered to match its citations."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for number, (label, score, text) in enumerate(sources, start=1):
            st.markdown(f"**[{number}] {label}** - similarity {score:.3f}")
            st.caption(text)


def render_trace(trace: QueryTrace) -> None:
    """How the question was handled. Off by default, invaluable when wrong."""
    if not st.session_state.show_debug:
        return

    with st.expander("How this was answered"):
        intent, route = trace.intent, trace.route
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Intent** `{intent.name}`")
            st.caption(f"decided by {intent.method} - {intent.reason}")
        with right:
            st.markdown(f"**Route** `{route.name}`")
            st.caption(route.reason)

        if trace.rewrite and trace.rewrite.changed:
            st.markdown("**Searched for instead**")
            st.caption(f"{trace.rewrite.original!r} -> {trace.rewrite.query!r}")

        for note in trace.notes:
            st.warning(note, icon=":material/info:")

        stages = " · ".join(f"{k} {v*1000:.0f}ms" for k, v in trace.timings.items())
        st.caption(
            f"{trace.api_calls} API call{'' if trace.api_calls == 1 else 's'} · "
            f"{stages or 'no timed stages'} · {trace.total_seconds:.2f}s total"
        )


def clear_index() -> None:
    for filename in (INDEX_FILE, METADATA_FILE):
        (config.INDEX_DIR / filename).unlink(missing_ok=True)
    st.session_state.store = None
    st.session_state.load_error = None


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
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

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
            clear_index()
            st.session_state.history = []
            st.rerun()
    elif st.session_state.get("load_error"):
        # The index exists but could not be opened, so the usual Clear button
        # above is out of reach - offer it here or there is no way out.
        if st.button("Delete the unreadable index"):
            clear_index()
            st.rerun()
    else:
        st.info(
            "No documents indexed yet. You can still ask general questions - "
            "upload a file to ask about your own."
        )

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

    st.divider()
    st.session_state.show_debug = st.toggle(
        "Show how each answer was reached",
        value=st.session_state.show_debug,
        help="Intent, route, the query actually searched for, timings and API calls.",
    )

    if st.session_state.history and st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()


# --- Main ------------------------------------------------------------------
st.title("Ask your documents")
st.caption(
    "Questions about your uploaded files are answered from them, with every claim "
    "cited back to its page. Anything else is answered directly."
)

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        render_sources(entry["sources"])
        if entry.get("trace"):
            render_trace(entry["trace"])

placeholder = (
    "Ask anything - about your documents, or not"
    if provider_ready
    else f"Set up {provider.name} first - see the sidebar"
)
question = st.chat_input(placeholder, disabled=not provider_ready)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Working out how to answer..."):
            plan = process(
                question,
                store=store,
                history=st.session_state.history,
                provider=provider,
            )

        answer = None
        if plan.answer is not None:
            # Known without a model - arithmetic.
            st.markdown(plan.answer)
            answer = plan.answer
        elif plan.message is not None:
            st.info(plan.message)
        else:
            try:
                answer = st.write_stream(plan.stream())
            except (ProviderNotReady, ProviderError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The request failed: {exc}")

        sources = [(chunk.label, score, chunk.text) for chunk, score in plan.sources]
        render_sources(sources)
        render_trace(plan.trace)

        if answer:
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "trace": plan.trace,
                }
            )
