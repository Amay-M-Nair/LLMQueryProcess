"""Azriel - the web page. Run it with:  streamlit run app.py

Every Streamlit call in the project is in this file, and no decision is. What
happens to a question is decided in backend/query_processor.py; this renders
the result.

The styling below is deliberately thin: a serif wordmark, hairline rules, and
six colour tokens per theme in PALETTES. Streamlit cannot change theme while
running, so dark mode is those tokens rendered as CSS over the base theme in
.streamlit/config.toml.
"""

import os
import re

import streamlit as st

from backend.llm import PROVIDERS, ProviderError, ProviderNotReady, get_provider, model_for
from backend.query_processor import QueryTrace, process
from ingestion.pipeline import build_index
from utils import config, env_file
from vectorstore.faiss_store import INDEX_FILE, VectorStore
from vectorstore.metadata_store import METADATA_FILE

APP_NAME = "Azriel"

st.set_page_config(
    page_title=APP_NAME,
    layout="centered",
    initial_sidebar_state="expanded",
)

# Streamlit cannot change theme at runtime, so the palette is applied as CSS
# over the top of the one in .streamlit/config.toml. Everything below reads
# from these six tokens - to retheme the app, change them and nothing else.
PALETTES = {
    "light": {
        "bg": "#fbfbfa",
        "panel": "#f4f4f1",
        "text": "#1c2427",
        "muted": "#6b7478",
        "rule": "#e3e3df",
        "accent": "#2f3e46",
    },
    "dark": {
        "bg": "#14181a",
        "panel": "#1b2023",
        "text": "#e8e6e1",
        "muted": "#8d9599",
        "rule": "#2a3134",
        "accent": "#cfd8d5",
    },
}


# One number for the size of the whole interface. Everything below is in rem,
# so raising the root font size scales the type, the spacing and the width of
# the reading column together. Better than zoom, which blurs on some displays
# and leaves fixed-position elements behind.
UI_SCALE = 1.25


def apply_theme(mode: str) -> None:
    """Paint the interface. Presentation only - nothing here reads state."""
    c = PALETTES[mode]
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&display=swap');

          html {{ font-size: {UI_SCALE * 100:.0f}%; }}

          /* Hide the chrome itself, never the toolbar that contains it.
             Streamlit renders the button that reopens a collapsed sidebar
             inside stToolbar, so display:none there makes collapsing the
             sidebar a one-way door. */
          [data-testid="stAppDeployButton"],
          [data-testid="stMainMenu"], #MainMenu,
          [data-testid="stToolbarActions"],
          [data-testid="stDecoration"],
          footer {{ display: none; }}

          [data-testid="stExpandSidebarButton"] button {{ color: {c['muted']}; }}
          [data-testid="stExpandSidebarButton"] button:hover {{ color: {c['text']}; }}

          /* Set the inherited colour too, so anything not styled below still
             lands the right way up in dark mode. */
          [data-testid="stAppViewContainer"],
          [data-testid="stHeader"] {{ background: {c['bg']}; color: {c['text']}; }}
          [data-testid="stSidebar"] {{
            background: {c['panel']};
            border-right: 1px solid {c['rule']};
            /* Streamlit fixes the sidebar width with an inline style, which
               no stylesheet rule can beat, and !important would win by
               breaking the drag-to-resize handle. Raising the floor scales
               the panel with the text and still lets it be dragged wider. */
            width: {300 * UI_SCALE:.0f}px !important;
          }}

          .block-container {{ padding-top: 3.5rem; max-width: 44rem; }}

          /* Streamlit styles headings specifically enough to win a plain
             class selector, so these are forced. Georgia is the fallback
             rather than a hope: Fraunces comes from Google Fonts and may not
             load at all behind a proxy or offline. */
          h1.azriel-mark, .stMarkdown h1.azriel-mark {{
            font-family: 'Fraunces', Georgia, 'Times New Roman', serif !important;
            font-size: 2.1rem !important;
            font-weight: 500 !important;
            line-height: 1.1 !important;
            letter-spacing: 0.01em !important;
            padding: 0 !important;
            margin: 0 0 0.15rem 0 !important;
            color: {c['text']} !important;
          }}
          .azriel-sub {{
            font-size: 0.85rem !important;
            color: {c['muted']} !important;
            margin: 0 0 1.1rem 0 !important;
          }}
          .azriel-rule {{
            border: 0;
            border-top: 1px solid {c['rule']};
            margin: 0 0 2rem 0;
          }}

          body, p, li, span, label, h1, h2, h3, h4,
          .stMarkdown, [data-testid="stChatMessageContent"] {{ color: {c['text']}; }}
          [data-testid="stCaptionContainer"], .stCaption, small {{ color: {c['muted']} !important; }}

          /* Streamlit reserves the sidebar header for a logo and leaves 60px
             of nothing when there isn't one. The wordmark goes there, which
             balances the collapse arrow sitting opposite it. */
          [data-testid="stSidebarHeader"] {{
            /* Left padding is zero because the spacer already carries a 20px
               inset, which is exactly where the headings below start. */
            padding: 0 1rem 0 0 !important;
            align-items: center;
          }}
          [data-testid="stLogoSpacer"] {{
            width: auto !important;
            height: auto !important;
            display: flex;
            align-items: center;
          }}
          [data-testid="stLogoSpacer"]::before {{
            content: "Azriel";
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.05rem;
            font-weight: 500;
            letter-spacing: 0.01em;
            color: {c['muted']};
          }}

          [data-testid="stSidebar"] h2 {{
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {c['muted']};
            margin-bottom: 0.4rem;
          }}
          [data-testid="stSidebar"] .stButton button {{ width: 100%; }}

          .stButton button, .stDownloadButton button {{
            background: transparent;
            color: {c['text']};
            border: 1px solid {c['rule']};
          }}
          .stButton button:hover {{ border-color: {c['accent']}; color: {c['accent']}; }}

          input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] {{
            background: {c['bg']} !important;
            color: {c['text']} !important;
            border-color: {c['rule']} !important;
          }}

          [data-testid="stChatMessage"] {{
            background: transparent;
            padding: 0.3rem 0 1rem 0;
          }}
          [data-testid="stChatInput"] {{
            background: {c['bg']};
            border: 1px solid {c['rule']};
          }}
          [data-testid="stChatInput"] textarea {{ color: {c['text']} !important; }}

          [data-testid="stExpander"] {{
            border: 1px solid {c['rule']};
            border-radius: 6px;
            background: transparent;
          }}
          [data-testid="stExpander"] summary {{ color: {c['muted']}; }}

          [data-testid="stFileUploaderDropzone"] {{
            background: {c['bg']};
            border: 1px dashed {c['rule']};
          }}

          /* The base theme in .streamlit/config.toml is dark, because that is
             what loads first and a white flash is worse than none. Streamlit
             paints its own widget chrome from that base, so switching to the
             light palette leaves dark buttons behind light text. Every widget
             Streamlit colours itself has to be repainted here, or it turns
             invisible in one theme or the other. */
          [data-testid^="stBaseButton"],
          [data-testid="stFileUploaderDropzone"] button,
          [data-baseweb="input"] button,
          [data-baseweb="base-input"] button,
          [data-testid="stSidebarCollapseButton"] button,
          [data-testid="stExpandSidebarButton"] button {{
            background: {c['bg']} !important;
            color: {c['text']} !important;
            border-color: {c['rule']} !important;
          }}
          [data-testid^="stBaseButton"]:hover {{ border-color: {c['accent']} !important; }}
          [data-testid="stBaseButton-primary"] {{
            background: {c['accent']} !important;
            color: {c['bg']} !important;
          }}
          /* A disabled primary button keeping its filled background put muted
             text on the accent colour, which is nearly unreadable. Disabled
             means it stops looking like the thing you should press. */
          [data-testid="stBaseButton-primary"]:disabled {{
            background: transparent !important;
            color: {c['muted']} !important;
            border-color: {c['rule']} !important;
          }}
          [data-baseweb="select"] div, [data-baseweb="popover"] li {{
            color: {c['text']} !important;
          }}
          /* Streamlit's material icons carry their own colour on an inner
             span, so setting it on the button they sit in does not reach
             them: the collapse arrow and the show-password eye stayed the
             base theme's colour and vanished in the other one. */
          [data-testid="stIconMaterial"] {{ color: {c['muted']} !important; }}
          button:hover [data-testid="stIconMaterial"] {{ color: {c['text']} !important; }}
          /* Streamlit hard-codes several colours for the light theme. Each of
             these leaked dark-on-dark or a light panel into the dark palette,
             so they are forced rather than merely set. */
          code, .stCode, [data-testid="stCode"], pre {{
            font-size: 0.85em !important;
            color: {c['accent']} !important;
            background: transparent !important;
          }}

          .stButton button p, .stButton button span {{ color: inherit !important; }}
          .stButton button:disabled,
          .stButton button:disabled p {{ color: {c['muted']} !important; opacity: 0.55; }}

          [data-testid="stFileUploaderDropzone"],
          [data-testid="stFileUploaderDropzone"] span,
          [data-testid="stFileUploaderDropzone"] small,
          [data-testid="stFileUploaderDropzone"] div {{ color: {c['muted']} !important; }}
          [data-testid="stFileUploaderDropzone"] button {{
            color: {c['text']} !important;
            border-color: {c['rule']} !important;
          }}

          [data-testid="stWidgetLabel"] p {{ color: {c['muted']} !important; }}
          [data-baseweb="popover"] li {{ color: {c['text']}; }}

          .azriel-cite {{
            font-size: 0.68em;
            font-weight: 600;
            color: {c['muted']};
            padding: 0 0.12em;
            vertical-align: super;
          }}
          .azriel-hint {{
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {c['muted']};
            margin: 1.5rem 0 0.6rem 0;
          }}

          /* Set the prose for reading rather than for filling the window. */
          [data-testid="stChatMessageContent"] p,
          [data-testid="stChatMessageContent"] li {{
            line-height: 1.62;
            font-size: 0.97rem;
          }}
          [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
            font-weight: 500;
          }}

          hr {{ border-color: {c['rule']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# What the status line settles on once the route is known. The point of
# saying it afterwards is that the user learns the routing exists.
SUMMARIES = {
    "retrieval": "Answered from your documents",
    "direct": "Answered directly",
    "calculate": "Worked out locally",
    "clarify": "Needs rephrasing",
}

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def decorate_citations(text: str) -> str:
    """Turn [1] and [2, 4] into something that reads as a reference mark.

    Plain brackets in running prose look like a typo or an unrendered link.
    Setting them small and raised marks them as apparatus rather than
    sentence, which is what they are.
    """
    if not text:
        return text
    # One mark per group, with separators inside it. Rendering [1, 2, 3] as
    # three adjacent superscripts produces "123", which reads as a number.
    return CITATION_RE.sub(
        lambda m: '<sup class="azriel-cite">'
        + ",".join(n.strip() for n in m.group(1).split(","))
        + "</sup>",
        text,
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
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

apply_theme(st.session_state.theme)

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

    # Three places a key can come from, and the sidebar says which is in use:
    # typed here (this session only), saved in .env (every session), or
    # nothing. A deployed copy relies on the first; your own machine is
    # better served by the second.
    if get_provider(provider_name).needs_key:
        variable = get_provider(provider_name).key_variable
        from_env = bool(os.environ.get(variable))
        typed = keys.get(provider_name, "")

        if st.session_state.pop("key_saved", None):
            st.success(f"`{variable}` saved to .env.")

        # The widget keeps whatever was typed across reruns, so emptying the
        # session dict is not enough to clear the box - saving would leave the
        # key sitting there and the sidebar still claiming it was typed rather
        # than stored. Bumping the key builds a fresh, empty widget instead.
        nonce = st.session_state.get("key_input_nonce", 0)
        entered = st.text_input(
            "API key",
            value=typed,
            type="password",
            key=f"api_key_{provider_name}_{nonce}",
            placeholder="Paste a key to use it here" if from_env else f"Paste your {variable}",
            help=(
                f"Used instead of {variable} for this session. Save it to keep "
                "it across restarts."
            ),
        )
        if entered != typed:
            keys[provider_name] = entered
            typed = entered

        if typed:
            st.caption("Using the key you entered.")
        elif from_env:
            st.caption(f"Using `{variable}` from your .env file.")
        else:
            st.caption(f"No key yet. Paste one above, or set `{variable}` in .env.")

        # "Ready" below only means a key is present. A key can be present and
        # be a typo, expired, or out of quota - all of which look identical
        # until the first question fails. This spends one request to find out,
        # on a button so it is never spent behind your back.
        if st.button("Verify key", help="Spends one API call.", disabled=not (typed or from_env)):
            candidate = get_provider(provider_name, api_key=typed or None)
            with st.spinner("Verifying"):
                try:
                    candidate.complete("Reply with the single word OK.", "Ready?")
                    st.success("Key accepted.")
                except Exception as exc:
                    st.error(f"Rejected: {exc}")

        if typed:
            if st.button("Save to .env", help="Keeps this key across restarts."):
                try:
                    env_file.set_value(variable, typed)
                except env_file.EnvWriteError as exc:
                    st.error(str(exc))
                else:
                    keys.pop(provider_name, None)   # it lives in .env now
                    st.session_state.key_input_nonce = nonce + 1
                    st.session_state.key_saved = True
                    st.rerun()

            if st.button("Discard", help="Forget the key you typed."):
                keys.pop(provider_name, None)
                st.session_state.key_input_nonce = nonce + 1
                st.rerun()

        elif from_env:
            if st.button("Remove from .env", help=f"Deletes {variable} from the file."):
                env_file.clear_value(variable)
                st.rerun()

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
    dark = st.toggle("Dark", value=st.session_state.theme == "dark")
    chosen = "dark" if dark else "light"
    if chosen != st.session_state.theme:
        # Repaint on the next run; restyling the current one would leave the
        # already-rendered sidebar in the old palette.
        st.session_state.theme = chosen
        st.rerun()

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
    f'<h1 class="azriel-mark">{APP_NAME}</h1>'
    '<p class="azriel-sub">Answers from your documents, cited. '
    'Everything else, answered directly.</p>'
    '<hr class="azriel-rule">',
    unsafe_allow_html=True,
)


def render_turn(entry: dict) -> None:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(decorate_citations(entry["answer"]), unsafe_allow_html=True)
        render_sources(entry["sources"])
        if entry.get("trace"):
            render_trace(entry["trace"])


for entry in st.session_state.history:
    render_turn(entry)

# --- Empty state -----------------------------------------------------------
# A blank column and a text box tell a first-time visitor nothing, least of
# all that this answers questions having nothing to do with their files. One
# example per route says it faster than a paragraph would.
has_docs = bool(store and len(store) > 0)
asked = st.session_state.pop("pending_question", None)

if not st.session_state.history and not asked and provider_ready:
    st.markdown('<p class="azriel-hint">Try</p>', unsafe_allow_html=True)
    examples = [
        (f"What does {store.sources[0]} say about my experience?", "from your documents")
        if has_docs else
        ("Upload a file to ask about your own documents", None),
        ("Explain what a vector embedding is", "general knowledge"),
        ("What is 15% of 240?", "worked out locally, no API call"),
    ]
    for label, note in examples:
        if note is None:
            st.caption(label)
            continue
        if st.button(label, key=f"eg-{label[:20]}"):
            st.session_state.pending_question = label
            st.rerun()
        st.caption(note)

placeholder = (
    "Ask a question"
    if provider_ready
    else f"Configure {provider.name} in the sidebar"
)
typed = st.chat_input(placeholder, disabled=not provider_ready)
question = typed or asked

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        # st.status carries the stage names out of the orchestrator, so the
        # several seconds spent deciding how to answer look like progress
        # rather than a frozen page.
        status = st.status("Reading your question", expanded=False)
        try:
            plan = process(
                question,
                store=store,
                history=st.session_state.history,
                provider=provider,
                on_stage=lambda label: status.update(label=label),
            )
        except Exception as exc:
            status.update(label="Could not work out how to answer", state="error")
            st.error(str(exc))
            st.stop()

        route = plan.trace.route.name
        status.update(label=SUMMARIES.get(route, route), state="complete")

        answer = None
        if plan.answer is not None:
            # Known without a model - arithmetic.
            st.markdown(plan.answer)
            answer = plan.answer
        elif plan.message is not None:
            st.info(plan.message)
        else:
            slot = st.empty()
            answer = ""
            try:
                for piece in plan.stream():
                    answer += piece
                    slot.markdown(answer)
                # Re-render once complete so the citation markers can be
                # styled; doing it per chunk would fight the stream.
                slot.markdown(decorate_citations(answer), unsafe_allow_html=True)
            except (ProviderNotReady, ProviderError) as exc:
                slot.error(str(exc))
                answer = None
            except Exception as exc:
                slot.error(f"The request failed: {exc}")
                answer = None

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
