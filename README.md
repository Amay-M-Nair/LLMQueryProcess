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

## Collections

Documents are kept in **collections**. A collection is a namespace with its
own index, and no collection can retrieve another's documents.

On one machine that is invisible: everything goes in `default`, because every
visitor is you. On a URL it is the whole ballgame - the page gives each
browser session a collection of its own, so one visitor's question can never
retrieve and cite another visitor's files. See *Deploying it* below.

Collection names become directory names, so they are checked rather than
trusted - letters, digits, dashes and underscores only. A name like
`../../etc` gets a rejection, not a path.

The page does not import the pipeline. It holds a client, and the client
either calls the pipeline in the same process or reaches a remote service
over HTTP; nothing above that line knows which. In-process is the default and
the only one that ships here. The seam exists so the UI depends on an
interface rather than on the pipeline itself.

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
| | `logs.py` | Where the detail goes when the user gets a sentence |
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

## Where LangChain is used, and where it is not

Two pieces of the pipeline are LangChain's, because they are strictly better
or exactly equal and save writing code that has already been written:

| | |
|---|---|
| `RecursiveCharacterTextSplitter` | Given a word-counting length function, so the limit it respects is the embedding model's token budget rather than a character count. It prefers a paragraph break, then a sentence, then a word - the fixed window it replaced closed at exactly 160 words wherever that fell, ending chunks "...before 10:00" and "...A". |
| `HuggingFaceEmbeddings` | Wraps the same sentence-transformers model and returns bit-for-bit identical vectors. Checked, not assumed. |

**Measured after the swap: 24/24 hit@2, MRR 1.000 - the same as before.** The
tidier chunk boundaries did not change what gets retrieved on a corpus this
size, so they are a readability win rather than an accuracy one, and are not
claimed as more than that.

Three other pieces stayed hand-written, each because LangChain's equivalent
would have cost something already built and tested:

- **BM25.** `BM25Retriever` returns documents with no score attached. Ranking
  here blends BM25 against cosine, which needs a number per chunk, and the
  stopword handling that stops a rare "the" dragging unrelated chunks up the
  ranking has no equivalent.
- **The providers.** `ChatGoogleGenerativeAI` takes one model. This falls
  through five when the daily free-tier quota on one is exhausted, reads the
  `retryDelay` the API supplies rather than guessing, and says which quota was
  hit. That is the difference between the free tier being usable and not.
- **The vector store.** The index is paired with a metadata store that
  identifies documents by content hash and removes one document's chunks
  without disturbing the rest.

`langchain-community` is avoided entirely: it prints a deprecation notice
saying it is being sunset and no longer maintained. The standalone
`langchain-*` packages are used instead.

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

## When something goes wrong

Every failure is reported twice: once to you, in language that says what to
do, and once to the log, with whatever the exception actually said. The
second half matters because the first was written to be reassuring rather
than diagnostic, and because several failures are swallowed deliberately - a
rewrite that fails falls back to the original question, an OCR page that
fails is skipped - and would otherwise leave no trace at all.

Logs go to stderr and to `data/azriel.log`. Set `LOG_FILE = None` in
`utils/config.py` to keep them on stderr alone.

**Two things never reach them: API keys, and the contents of your documents.**
Keys because logs get pasted into issues; documents because a log file is not
where your private files belong. Both are covered by tests that plant a
secret and assert it does not appear - which is how the one real leak here
was found. A provider whose error message carried the key in a URL had that
message copied into the intent's `reason`, which is logged *and* shown in the
debug panel. Redaction now happens at the logging filter, so nothing reaches
a handler unredacted, and again where the reason is built, so the panel is
covered too.

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

### The one thing to get right

On your own machine every visitor is you, so one shared index is correct and
your documents are still there tomorrow. On a URL it is a document leak:
everyone reads everyone else's files, and the answers cite them.

So the page has two modes, and hosting it means picking the second:

```
AZRIEL_COLLECTIONS=visitor   a fresh collection per browser session
AZRIEL_PUBLIC=1              refuse to start if the above is not set
```

`AZRIEL_PUBLIC` exists because the failure is silent. A deployment left on
shared looks perfectly normal until two people use it, and nothing on the
page says otherwise - so it stops rather than serves. Both are lines of
the secrets you paste in when deploying.

Visitor collections are forgotten when the tab closes. That is the trade:
uploading again after a refresh, in exchange for documents that are nobody
else's business.

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. share.streamlit.io -> **Create app** -> pick the repo, branch `main`,
   file `app.py`.
3. Open **Advanced settings** before deploying. Set the Python version to
   **3.11**, and paste into **Secrets**:

```toml
GOOGLE_API_KEY = "your-key"
AZRIEL_COLLECTIONS = "visitor"
AZRIEL_PUBLIC = "1"
```

Those three are read from the environment by `utils/config.py`, and
Community Cloud sets every root-level secret as an environment variable, so
nothing in the code has to know where it came from. Keep them quoted - the
public check is a string comparison against `"1"`.

The first build takes several minutes. The app then lives at
`https://<something>.streamlit.app`, and **sleeps after 12 hours** without
traffic; the next visitor wakes it.

Two files exist only for this host. `packages.txt` installs `libgl1` and
`libglib2.0-0`, which opencv needs and which OCR pulls in - without them a
scanned PDF is skipped instead of read. And `requirements.txt` asks for
torch from PyTorch's CPU index, because the wheel PyPI serves on Linux
bundles 2.5 GB of CUDA libraries for a GPU no free host has. That one
matters everywhere, not just here: it is most of the install.

The ceiling is 2.7 GB of memory and 2 CPU cores, shared. Loading the
embedding model and answering a question fits; what will not fit is a very
large index held in memory alongside it. If the app shows "over its resource
limits", that is what happened.

### What is still true after deploying

- **The filesystem is ephemeral.** For visitor collections that is the
  intent: documents go when the tab does. Nothing an app writes survives a
  reboot, so the index is always rebuilt from what visitors upload.
- **One free-tier key serves every visitor.** The sidebar lets someone bring
  their own, which is the only thing that scales past twenty questions a day.
- **The model reads figures but does not understand them** - see the limits
  below, because that one produces confident wrong answers rather than errors.

