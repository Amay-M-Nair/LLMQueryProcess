"""The web page. Run it with:  streamlit run app.py"""

import streamlit as st

from llmqp import answerer, config
from llmqp.ingest import build_index, retrieve
from llmqp.providers import (
    PROVIDERS,
    ProviderError,
    ProviderNotReady,
    get_provider,
    model_for,
)
from llmqp.store import VectorStore

st.set_page_config(page_title="Document Q&A", layout="wide")


def render_sources(sources: list[tuple[str, float, str]]) -> None:
    """Show the excerpts the answer was built from, numbered to match its citations."""
    with st.expander(f"Sources ({len(sources)})"):
        for number, (label, score, text) in enumerate(sources, start=1):
            st.markdown(f"**[{number}] {label}** - similarity {score:.3f}")
            st.caption(text)


# --- State -----------------------------------------------------------------
if "store" not in st.session_state:
    st.session_state.store = VectorStore.load(config.INDEX_DIR)
if "history" not in st.session_state:
    st.session_state.history = []
if "provider_name" not in st.session_state:
    st.session_state.provider_name = config.PROVIDER

store: VectorStore | None = st.session_state.store


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
            for name, reason in report.skipped:
                st.warning(f"Skipped **{name}** - {reason}")
            if report.indexed:
                st.rerun()
        except Exception as exc:
            status.update(label="Indexing failed", state="error")
            st.exception(exc)

    st.divider()

    if store and len(store) > 0:
        st.metric("Chunks indexed", len(store))
        st.caption("Files in the index:")
        for name in store.sources:
            st.caption(f"- {name}")

        if st.button("Clear index"):
            for filename in ("vectors.npy", "chunks.json"):
                (config.INDEX_DIR / filename).unlink(missing_ok=True)
            st.session_state.store = None
            st.session_state.history = []
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

    provider = get_provider(st.session_state.provider_name)
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
                "Try rephrasing, or lower MIN_SIMILARITY in llmqp/config.py."
            )
        else:
            answer = None
            try:
                answer = st.write_stream(
                    answerer.stream_answer(question, retrieved, provider=provider)
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
