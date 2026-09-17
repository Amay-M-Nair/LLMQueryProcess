"""Every tunable number in the project, in one place."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env so API keys land in the environment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
UPLOAD_DIR = DATA_DIR / "uploads"

# --- Retrieval (runs locally, costs nothing) -------------------------------
# A small, fast embedding model. 90 MB, downloads once on first use.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # output size of the model above

# MUST stay under what the embedding model can read in one go - all-MiniLM
# truncates at 256 tokens, about 175 words. A larger chunk is not an error and
# not a warning: the overflow is simply never seen by the search, so those
# words can never be found. build_index() checks this and tells you.
CHUNK_WORDS = 160
CHUNK_OVERLAP_WORDS = 40   # overlap so a sentence split across chunks isn't lost
TOP_K = 5                  # how many chunks to hand the model

# Cosine similarity below this counts as "not actually related". Without a
# floor the store always returns its k closest chunks even when nothing in
# the index is relevant, which invites the model to answer from noise.
MIN_SIMILARITY = 0.12

# If the whole index is smaller than this, skip retrieval and send everything.
# A short document does not need compressing, and retrieving over one can only
# lose information. Raise it if your model has a large context window.
FULL_CONTEXT_WORDS = 4000

# How much keyword (BM25) matching counts against vector similarity when
# ranking. 0.0 = pure meaning, 1.0 = pure keyword. Keywords matter for lookups
# like a name or an error code; meaning matters for everything else.
KEYWORD_WEIGHT = 0.35

# --- Which model writes the answer -----------------------------------------
# "gemini"    free tier, needs GOOGLE_API_KEY  -> https://aistudio.google.com/apikey
# "ollama"    free and offline, needs Ollama running locally
# "anthropic" paid, needs ANTHROPIC_API_KEY
PROVIDER = "gemini"

TEMPERATURE = 0.2  # low: we want the model sticking to the excerpts

# Verified working on 2026-09-15. Free-tier models get retired and go busy,
# so if the first is unavailable the provider falls through this list.
GEMINI_MODEL = "gemini-3.6-flash"
# The free tier allows only ~20 requests PER DAY PER MODEL, so each extra
# model in this list adds roughly another 20 questions a day. The provider
# falls through them in order when one is exhausted or overloaded.
GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]
GEMINI_MAX_TOKENS = 4096
GEMINI_RETRY_ROUNDS = 3      # passes over the model list before giving up
GEMINI_RETRY_BACKOFF = 2.0   # seconds, used only if the API suggests nothing
GEMINI_MAX_WAIT = 45.0       # never block the UI longer than this between tries

OLLAMA_MODEL = "llama3.2:3b"     # fits comfortably in 4 GB of VRAM
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_NUM_PREDICT = 2048
OLLAMA_TIMEOUT = 300             # seconds; local models can be slow

ANTHROPIC_MODEL = "claude-opus-5"
ANTHROPIC_EFFORT = "medium"      # low | medium | high | xhigh | max
ANTHROPIC_MAX_TOKENS = 16000     # thinking tokens count toward this
USE_REFUSAL_FALLBACKS = True     # retry on another model if a request is declined

# --- The prompt ------------------------------------------------------------
SYSTEM_PROMPT = """You answer questions using ONLY the numbered source excerpts provided.

Rules:
- Every factual claim must be followed by a citation like [1] or [2, 4] naming the
  excerpt(s) it came from.
- If the excerpts do not contain the answer, say so plainly. Do not use outside
  knowledge to fill the gap, and do not guess.
- If the excerpts conflict, say that and cite both sides.
- Be concise and direct. No preamble."""
