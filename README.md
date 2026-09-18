# Azriel

Ask questions about your own PDFs and notes. Answers come only from the files
you upload, and every claim is cited back to the file and page it came from.

This is a **RAG** system (retrieval-augmented generation): the standard way to
make a language model answer questions about private documents it was never
trained on.

## How it works

```
  your files
      |
 [document_loader]   read PDFs, Word files and text into pages,
      |              running OCR over any page that has no text layer
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

## What happens to a question

Not every question needs the same treatment, and the expensive paths are the
ones worth avoiding. A question is classified first, then routed:

| Intent | Route | Costs |
|---|---|---|
| `calculation` | worked out locally with an AST walk | **nothing** |
| `general` | straight to the model, no retrieval | one call |
| `document_query` | retrieval, answer cited back to the page | one or two calls |
| `summarization` | retrieval over the named document | one or two calls |
| `unknown` | asks you to rephrase | nothing |

Classification tries rules before it tries the model. Arithmetic, summary
requests, empty input and "nothing is indexed" are all settled locally, so the
common question costs one API call rather than three — which on a tier
allowing twenty requests a day is the difference between working all day and
being locked out by lunchtime.

Follow-ups are rewritten before retrieval, but only when they look dependent
on the conversation — a dangling pronoun, a continuation opener, a very short
question. "What about its limitations?" is searched for as "What are the
limitations of Transformers?", while the answer is still generated from what
you actually typed.

Turn on **Show how each answer was reached** in the sidebar to see the intent,
the route, the query that was actually searched for, the timings and the
number of API calls spent.

### Does any of it help?

`evaluation/` holds a corpus whose facts are known by construction, 35
labelled questions across the five intents, and a scorer. Run it with:

```bash
.venv/Scripts/python.exe -m evaluation.run_eval --answers
```

Against Gemini, over 15 chunks:

| | |
|---|---|
| Intent accuracy | 32/32 — 6/6 by rule, 26/26 by model |
| Retrieval hit@2 | 25/25, MRR 1.000 |
| Answer contains the expected fact | 32/32 |
| Citations that resolve to a real excerpt | 35/35 |
| Absent facts admitted as absent | 2/2 |
| API calls | 1.71 per question; 4 of 35 answered for nothing |
| Latency | median 6.4s, classification 2.6s of it |

Two caveats, because a number is only worth the test behind it. **The corpus
is 15 chunks.** Retrieval scores are reported alongside the share of the index
returned per question, and the harness prints a warning when that share is
large enough for hit@k to be unable to fail — at `--top-k 5` it does, at
`--top-k 2` it does not, which is why the table quotes hit@2. Nothing here
predicts behaviour over thousands of chunks. **And the dataset is small
enough to overfit to**, so `evaluation/dataset.jsonl` keeps the cases that
have failed, including one marked `KNOWN HARD` where the classifier does not
reliably follow its own instruction.

### Where an answer is allowed to come from

Document questions are answered from the excerpts and cite them. Where the
excerpts do not cover the question, the answer says so and may then continue
under a **Beyond your documents** heading, which carries no citations. A
marked answer is more use than a dead end, and the boundary stays visible.

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

### What it can read

`.pdf`, `.docx`, `.txt` and `.md`.

A **scanned PDF** has no text layer - it is pictures of words - so its pages
are rasterised and read back with OCR. That costs a few seconds a page, so it
only happens to pages that yielded nothing on their own: a report that is
typed with a scanned appendix only pays for the appendix. The OCR
dependencies are about 47 MB and are optional; without them a scan is skipped
with an explanation rather than silently indexed as empty.

A **.docx** has no pages of its own - pagination belongs to whatever renders
the file - so only page breaks the author inserted start a new page, and a
document without them is page 1. Tables are read as well as paragraphs,
because policy documents put their most citable facts in tables.

Run the tests with:

```bash
.venv/Scripts/python.exe -m pytest
```

## Project layout

The code is split into four layers. Each depends only on the layers below it,
so any single piece can be replaced without disturbing the rest.

| Layer | Module | Job |
|---|---|---|
| UI | `app.py` | The web page — all Streamlit code lives here and nowhere else |
| | `.streamlit/config.toml` | Theme colours, changeable without reading any code |
| **backend** | `intent_classifier.py` | What kind of question is this? |
| | `router.py` | Which path should it take? |
| | `calculator.py` | Arithmetic, worked out locally |
| | `rag.py` | Builds the prompt, hands it to the chosen provider |
| | `llm.py` | Provider registry — picks one by name |
| | `providers/` | One adapter per answer model (Gemini, Ollama, Claude) |
| **ingestion** | `document_loader.py` | File → pages of text, OCR included |
| | `chunker.py` | Pages → overlapping chunks |
| | `embedder.py` | Text → vectors |
| | `pipeline.py` | Ties the ingest together; `build_index` and `retrieve` |
| **vectorstore** | `faiss_store.py` | The FAISS index: saving, loading, similarity search |
| | `metadata_store.py` | Chunk records and the register of indexed documents |
| | `keyword.py` | BM25 keyword scoring, blended into the ranking |
| **utils** | `config.py` | Every tunable number in one place |
| | `preprocessing.py` | Tidies a question without changing what it asks |
| | `prompts.py` | Every prompt the system sends |
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
- **`KEYWORD_RESCUE`** (0.6) — how good a keyword match has to be to survive
  `MIN_SIMILARITY` on its own. A chunk naming someone once is a real answer to
  a question about them, however low its cosine.
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

## Why FAISS and not pgvector

| | FAISS | pgvector |
|---|---|---|
| Setup | a file under `data/index/` | a running PostgreSQL server |
| Accuracy at this scale | `IndexFlatIP` is **exact**, sub-millisecond for thousands of chunks | HNSW/IVFFlat are **approximate** — trading exactness for speed this corpus does not need |
| Hybrid keyword search | BM25 blended into the ranking, already built | Postgres full-text uses `ts_rank`, **not** BM25 — the blend would need rewriting |
| Filtered search | needs an `IDSelector`, or over-fetch then filter | native: `WHERE user_id = $2 ORDER BY embedding <=> $1` |
| Concurrent writers | single-writer index files | transactional |

pgvector is the right answer for multiple users, deployed, with filtered
search; the wrong one for a single person on a laptop with a few hundred
chunks. Swapping later is a small job — everything goes through `VectorStore`,
so it means one new module in `vectorstore/` and no changes anywhere else.

## Known limits

- **Charts and diagrams are read as loose text, not understood.** OCR recovers
  the words and numbers in a figure but not its layout, and which bar a number
  sits above is layout. A bar chart of Q1-Q4 came back as `240, 185, 150, 120`
  and `Q1, Q2, Q3, Q4` on separate lines - every value correct, every pairing
  lost. Asked for the Q1 figure, the answer was "240 refunds in Q1 [1]" when
  the truth was 120, and it carried a citation, which makes a wrong number look
  checked. Reading the page with a vision model instead of OCR would fix it;
  until then, treat any answer drawn from a figure as unverified.
- **No reranking.** Ranking is vector similarity blended with BM25. A
  cross-encoder reranker over the top 20 would improve it further.
- **Questions about a document, in a large corpus.** "Who wrote this?" works
  only when the whole index fits under `FULL_CONTEXT_WORDS`. Above that, such a
  question can fail: if its words appear nowhere in the text, neither vector
  nor keyword search can find the right chunk. Storing per-document summaries
  and searching those too would fix it.
- **One index, shared by everyone.** Fine on your own machine, disqualifying
  on a public URL - see below.

## Deploying it

Not done, and the reason is worth stating plainly rather than leaving as a
surprise.

### The blocker

`data/index` is a single index. Deployed publicly, every visitor's uploads
land in it, so one person's question retrieves and cites another person's
documents. That is a privacy leak, not a rough edge, and it is the thing to
fix before a link goes anywhere.

Streamlit cannot fix it: it has sessions but no users, and a document has to
belong to someone. Scoping documents to a person needs authentication, which
is where a web framework earns its place.

### What deployment changes about the design

Two decisions taken for local use invert once the app is hosted, which is
worth knowing before reading them as mistakes.

| | Local (what this is) | Deployed |
|---|---|---|
| Embeddings | local MiniLM - free, offline, no quota | **API embeddings**, which removes `torch` entirely |
| Vector store | a FAISS file | **pgvector**, so `WHERE user_id = ...` and the similarity search are one query |

Neither was wrong. Locally, API embeddings would exhaust a 20-a-day quota
before a single question could be asked. Hosted, with per-user data, `torch`
is 531 MB of dead weight in a web process and FAISS becomes one index file per
person to manage by hand. The [FAISS and pgvector](#why-faiss-and-not-pgvector)
comparison above describes exactly this case.

Dependency weight is the practical obstacle either way: `torch` 531 MB,
everything else about 90 MB, plus a 90 MB model download on first run.
Dropping local embeddings removes most of it.

### The shape it would take

```
Browser  ->  Django (auth, sessions, uploads, streaming views)
               |
             backend/ ingestion/ vectorstore/     <- unchanged
               |
             Postgres + pgvector   ·   Gemini API
```

**The layers already separate correctly.** `backend.query_processor.process()`
does not know Streamlit exists - it takes a question and returns a plan - so a
view would call it the same way `app.py` does:

```python
def ask(request):
    plan = process(
        request.POST["question"],
        store=store_for(request.user),
        history=request.session.get("history", []),
        provider=get_provider("gemini"),
    )
    return StreamingHttpResponse(plan.stream(), content_type="text/plain")
```

Only `app.py` is replaced.

### Django or FastAPI

The original plan named FastAPI, and for a thin JSON API it is the better fit:
lighter, async by default, OpenAPI documentation for free. Django is the
better fit here because the problem is not the API - it is users, sessions and
per-user storage, which `django.contrib.auth` and the ORM provide outright.
Pick FastAPI if the front end is separate and there is no login; pick Django if
accounts are the point.

### Steps

1. `pip install django psycopg[binary] pgvector gunicorn whitenoise`
2. `django-admin startproject azriel` beside the existing packages
3. Models: `Document(user, name, content_hash, pages)` and
   `Chunk(document, text, page, chunk_id, embedding=VectorField(384))`
4. `vectorstore/pgvector_store.py` implementing the same `add` / `search` /
   `remove_document` interface - nothing above it changes
5. An API-backed embedder behind the same `embed()` signature
6. Views for upload, ask (streaming), and history
7. `GOOGLE_API_KEY`, `SECRET_KEY` and `DATABASE_URL` from the environment
8. Render or Railway, both of which offer managed Postgres with pgvector

**`DEBUG = False` and a real `ALLOWED_HOSTS` before the link is shared.**
Django's debug pages print the environment, API keys included.

### If you only want a demo

Hosting the Streamlit app as-is is fine provided the link is yours alone.
Hugging Face Spaces suits it best - 16 GB of memory, native Streamlit support,
secrets in the settings rather than in `.env`. Streamlit Community Cloud is
the easier integration but its free memory limit is tight against 620 MB of
dependencies. Either way, the filesystem is ephemeral: the index is lost on
restart, and the shared-index problem above still applies.

