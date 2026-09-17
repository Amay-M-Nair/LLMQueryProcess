# LLM Query Processing & RAG

Ask questions about your own PDFs and notes. Answers come only from the files
you upload, and every claim is cited back to the file and page it came from.

This is a **RAG** system (retrieval-augmented generation): the standard way to
make a language model answer questions about private documents it was never
trained on.

## How it works

```
  your files
      |
 [document_loader]   read PDFs / text into pages
      |
    [chunker]        split each page into ~160-word overlapping windows
      |
   [embedder]        turn each chunk into a 384-number vector, locally
      |
    [store]          keep vectors + chunk metadata on disk
      |
  -----------------------------------------------------------------
      |
  your question  ->  embedded the same way
      |
 [store.search]      find the chunks whose vectors point the same direction,
      |              blended with BM25 keyword overlap
     [rag]           paste those chunks into a prompt, ask the model to answer
      |              using only them, with [1][2] citations
      v
   the answer
```

The one idea that makes it work: two pieces of text about the same subject get
vectors pointing in a similar direction, so "closest vector" means "most
related passage" — even when the question and the document share no words.

## Setup

1. Create the environment and install dependencies:

   ```bash
   python -m venv .venv
   ```

   ```bash
   .venv/Scripts/python.exe -m pip install -r requirements.txt
   ```

2. Add an API key. Copy `.env.example` to `.env` and paste in a **free**
   Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey):

   ```
   GOOGLE_API_KEY=...
   ```

   Check it worked:

   ```bash
   .venv/Scripts/python.exe -m utils.check
   ```

3. Run it:

   ```bash
   .venv/Scripts/streamlit.exe run app.py
   ```

   Your browser opens at `http://localhost:8501`. Upload files in the sidebar,
   click **Add to index**, then ask questions.

The first run downloads the embedding model (~90 MB), once.

## Project layout

The code is split into four layers. Each depends only on the layers below it,
so any single piece can be replaced without disturbing the rest.

| Layer | Module | Job |
|---|---|---|
| UI | `app.py` | The web page — all Streamlit code lives here and nowhere else |
| **backend** | `rag.py` | Builds the prompt, hands it to the chosen provider |
| | `llm.py` | Provider registry — picks one by name |
| | `providers/` | One adapter per answer model (Gemini, Ollama, Claude) |
| **ingestion** | `document_loader.py` | File → pages of text |
| | `chunker.py` | Pages → overlapping chunks |
| | `embedder.py` | Text → vectors |
| | `pipeline.py` | Ties the ingest together; `build_index` and `retrieve` |
| **vectorstore** | `faiss_store.py` | Saving, loading, and similarity search |
| | `keyword.py` | BM25 keyword scoring, blended into the ranking |
| **utils** | `config.py` | Every tunable number in one place |
| | `check.py` | `python -m utils.check` — tells you what is set up |

Because the UI is confined to `app.py`, you can drive the same pipeline from a
script, a notebook, or a CLI without touching any of it.

Generated files — the index, uploaded documents, your `.env` — all live under
`data/` and are not tracked.

## Choosing a provider

Retrieval always runs locally and is free. Only the answer step calls out, so
switching providers changes nothing else in the system.

| `PROVIDER` | Cost | Needs | Trade-off |
|---|---|---|---|
| `gemini` | Free tier | `GOOGLE_API_KEY` | **~20 requests/day per model**; documents go to Google |
| `ollama` | Free forever | Ollama running locally | Offline and private, but a 3B model follows the citation rules less reliably |
| `anthropic` | ~$0.008-0.04/question | `ANTHROPIC_API_KEY` | Costs money; best instruction-following |

Set it in `utils/config.py`, or pick it from the sidebar while the app is
running. To add a fourth, write a `Provider` subclass in `backend/providers/`
and add one line to the registry in `backend/llm.py`.

### The Gemini free-tier quota (important)

The free tier allows roughly **20 requests per day, per model** — the quota id
is `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. That is a *daily* cap,
not per-minute, and it is easy to hit while experimenting.

Because it is *per model*, `GEMINI_FALLBACK_MODELS` in `utils/config.py` is
effectively your daily budget: the provider falls through the list when a model
is exhausted or overloaded, so five models means roughly 100 questions a day.
Add more model names from `python -m utils.check --models` to extend it.

When a request is rejected the provider reads the `retryDelay` the API supplies
and waits exactly that long before retrying, rather than guessing.

If you run out for the day, switch the sidebar to `ollama` and keep working
offline.

### Running fully offline with Ollama

A 3B model fits in roughly 4 GB of VRAM:

```bash
ollama pull llama3.2:3b
```

Then set `PROVIDER = "ollama"`. Larger models will run but spill into system
RAM and slow to a crawl.

## Things worth tuning

All in `utils/config.py`:

- **`CHUNK_WORDS`** (160) — **must stay under ~176 words.** The embedding model
  reads at most 256 tokens; anything beyond that is silently truncated, so
  those words can never be found by search even though they are stored and
  shown to the model. `build_index()` warns if you exceed it.
- **`TOP_K`** (5) — more chunks means more context and more cost per question.
- **`MIN_SIMILARITY`** (0.12) — the floor below which a chunk counts as
  unrelated. Only applies to indexes larger than `FULL_CONTEXT_WORDS`.
- **`FULL_CONTEXT_WORDS`** (4000) — below this total size, retrieval is skipped
  and the whole index is sent. See below.
- **`KEYWORD_WEIGHT`** (0.35) — how much literal word overlap counts against
  meaning when ranking. Raise it for documents full of names, codes and IDs.
- **`PROVIDER`** (`gemini`) — which model writes the answer. Switchable in the
  sidebar at runtime too.

## Why small documents skip retrieval

Retrieval exists to fit a corpus that is too big for the context window into a
prompt. Run it over a two-page document and it can only lose information.

This is not theoretical. Asking a 556-word resume "what is the name of the
candidate?" failed under pure vector search: the best chunk scored **0.131**,
below the old 0.15 floor — while "best recipe for chocolate cake" scored
**0.070** against the same document. The relevant and irrelevant scores
*overlapped*, so no threshold could separate them. BM25 did not help either,
because the words "name" and "contact" appear nowhere in a resume.

So when the index is under `FULL_CONTEXT_WORDS`, everything is sent and the
model decides what is relevant. Retrieval only kicks in above that size.

## Known limits

- **General-knowledge questions do not work yet.** The chat box stays disabled
  until a document is indexed, and the prompt forbids answering from anything
  but the excerpts. Query routing is what fixes this.
- **Scanned PDFs do not work.** If a PDF is page images rather than text, there
  is nothing to extract; the app detects this and tells you. Fixing it needs an
  OCR step.
- **Re-indexing is by filename.** A file whose name is already in the index is
  skipped, even if its contents changed. Clear the index to force a rebuild.
- **No reranking.** Ranking is vector similarity blended with BM25. A
  cross-encoder reranker over the top 20 would improve it further.
- **Questions about a document, in a large corpus.** "Who wrote this?" works
  only when the whole index fits under `FULL_CONTEXT_WORDS`. Above that, such a
  question can fail: if its words appear nowhere in the text, neither vector
  nor keyword search can find the right chunk. Storing per-document summaries
  and searching those too would fix it.

## Where this is going

The layered structure above is groundwork for a query-processing layer that
does not exist yet. Planned, in order:

1. **FAISS index** with a proper metadata store, and content hashing so an
   edited document is actually re-indexed.
2. **Intent classification and routing** — send general questions straight to
   the model and document questions through retrieval, so both work.
3. **Query rewriting** — resolve follow-ups like "what about its limitations?"
   against the conversation so far.
4. **Evaluation** — a small labelled dataset measuring intent accuracy,
   retrieval relevance, answer faithfulness and latency, to show the query
   layer earns its keep.
