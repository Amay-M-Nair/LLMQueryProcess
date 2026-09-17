"""Azriel - the web page. Run it with:  streamlit run app.py

Every Streamlit call in the project is in this file, and no decision is. What
happens to a question is decided in backend/query_processor.py; this renders
the result.

The styling below is deliberately thin: a serif wordmark, one accent colour,
and hairline rules. Colours live in .streamlit/config.toml so they can be
changed without reading any of this.
"""

import os

import streamlit as st

from backend.llm import PROVIDERS, ProviderError, ProviderNotReady, get_provider, model_for
from backend.query_processor import QueryTrace, process
from ingestion.pipeline import build_index
from utils import config
from vectorstore.faiss_store import INDEX_FILE, VectorStore
from vectorstore.metadata_store import METADATA_FILE

APP_NAME = "Azriel"

st.set_page_config(
    page_title=APP_NAME,
    layout="centered",
    initial_sidebar_state="expanded",
)

# Restraint, mostly by removal: the toolbar, the footer, and the default
# heading weights. A reading surface should look like one.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500&display=swap');

      [data-testid="stToolbar"], footer, #MainMenu { display: none; }
      [data-testid="stDecoration"] { display: none; }

      .block-container { padding-top: 3.5rem; max-width: 46rem; }

      .azriel-mark {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        margin: 0 0 0.15rem 0;
        color: #1c2427;
      }
      .azriel-sub {
        font-size: 0.85rem;
        color: #6b7478;
        margin: 0 0 1.1rem 0;
        font-weight: 400;
      }
      .azriel-rule {
        border: 0;
        border-top: 1px solid #e3e3df;
        margin: 0 0 2rem 0;
      }

      /* Sidebar: quiet section labels rather than headings that shout. */
      [data-testid="stSidebar"] h2 {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #6b7478;
        margin-bottom: 0.4rem;
      }
      [data-testid="stSidebar"] .stButton button { width: 100%; }

      /* Chat: no bubbles, just indentation and a rule between turns. */
      [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.35rem 0 0.9rem 0;
      }

      .stExpander { border: 1px solid #e3e3df; border-radius: 6px; }
      code { font-size: 0.85em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sources(sources: list[tuple[str, float, str]]) -> None:
    """The excerpts an answer was built from, numbered to match its citations."""
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)}"):
        for number, (label, score, text) in enumerate(sources, start=1):
            st.markdown(f"**[{number}]** {label} · {score:.3f}")
            st.caption(text)


def render_trace(trace: QueryTrace) -> None:
    """How the question was handled. Off by default, invaluable when wrong."""
    if not st.session_state.show_debug:
        return

    with st.expander("Reasoning"):
        intent, route = trace.intent, trace.route
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Intent** `{intent.name}`")
            st.caption(f"decided by {intent.method} - {intent.reason}")
        with right:
            st.markdown(f"**Route** `{route.name}`")
            st.caption(route.reason)

        if trace.rewrite and trace.rewrite.changed:
            st.markdown("**Searched for**")
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
        "Add files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Index", type="primary", disabled=not uploads):
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for upload in uploads:
            destination = config.UPLOAD_DIR / upload.name
            destination.write_bytes(upload.getbuffer())
            saved_paths.append(destination)

        status = st.status("Indexing", expanded=True)
        try:
            new_store, report = build_index(
                saved_paths,
                existing=store,
                on_progress=lambda message: status.write(message),
            )
            new_store.save(config.INDEX_DIR)
            st.session_state.store = new_store
            store = new_store

            status.update(label=f"{report.chunks_added} passages added", state="complete")
            for name in report.reindexed:
                st.caption(f"{name} re-indexed - contents had changed")
            for name, reason in report.skipped:
                st.warning(f"{name} skipped - {reason}")
            if report.changed:
                st.rerun()
        except Exception as exc:
            status.update(label="Indexing failed", state="error")
            st.exception(exc)

    st.divider()

    if store and len(store) > 0:
        for name, record in store.documents.items():
            st.caption(f"{name} · {record.chunk_count}")
        st.caption(f"{len(store)} passages in total")

        if st.button("Clear"):
            clear_index()
            st.session_state.history = []
            st.rerun()
    elif st.session_state.get("load_error"):
        # The index exists but could not be opened, so the usual Clear button
        # above is out of reach - offer it here or there is no way out.
        if st.button("Delete unreadable index"):
            clear_index()
            st.rerun()
    else:
        st.caption("Nothing indexed. General questions still work.")

    st.divider()
    st.header("Model")

    provider_names = list(PROVIDERS)
    st.session_state.provider_name = st.selectbox(
        "Provider",
        provider_names,
        index=provider_names.index(st.session_state.provider_name),
        help="Retrieval runs locally. Only this step leaves the machine.",
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
        if st.button("Verify key", help="Spends one API call."):
            candidate = get_provider(provider_name, api_key=typed or None)
            with st.spinner("Verifying"):
                try:
                    reply = candidate.complete("Reply with the single word OK.", "Ready?")
                    st.success("Key accepted.")
                except Exception as exc:
                    st.error(f"Rejected: {exc}")

    provider = get_provider(provider_name, api_key=keys.get(provider_name) or None)
    st.caption(f"Model: `{model_for(provider.name)}`")
    st.caption(provider.note)

    try:
        provider.check_ready()
        st.caption(":material/check_circle: Ready")
        provider_ready = True
    except ProviderNotReady as exc:
        st.warning(str(exc))
        provider_ready = False

    st.divider()
    st.session_state.show_debug = st.toggle(
        "Show reasoning",
        value=st.session_state.show_debug,
        help="Intent, route, the query searched for, timings and API calls.",
    )

    if st.session_state.history and st.button("New conversation"):
        st.session_state.history = []
        st.rerun()


# --- Main ------------------------------------------------------------------
st.markdown(
    f'<p class="azriel-mark">{APP_NAME}</p>'
    '<p class="azriel-sub">Answers from your documents, cited. '
    'Everything else, answered directly.</p>'
    '<hr class="azriel-rule">',
    unsafe_allow_html=True,
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
    "Ask a question"
    if provider_ready
    else f"Configure {provider.name} in the sidebar"
)
question = st.chat_input(placeholder, disabled=not provider_ready)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
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
