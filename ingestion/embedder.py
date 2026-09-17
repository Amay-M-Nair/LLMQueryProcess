"""Turn text into vectors.

An embedding is a list of ~384 numbers representing a piece of text's meaning.
Two texts about the same thing end up with vectors pointing in a similar
direction, which is what makes search-by-meaning possible.

`sentence_transformers` pulls in torch and takes ~20s to import, so it is
imported inside `get_model()` rather than at module level - that keeps the web
app's startup instant and defers the cost to the first embed.
"""

from typing import TYPE_CHECKING

import numpy as np

from utils import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model = None


def get_model(name: str = config.EMBED_MODEL) -> "SentenceTransformer":
    """Load the embedding model once and reuse it (it is slow to construct).

    Once the weights are cached, loading is a local operation - except that
    sentence-transformers still contacts the HuggingFace hub to check for a
    newer revision. When the hub is slow or unreachable that check can hang
    for minutes with nothing to show for it, which looks exactly like the
    application having frozen. So the cache is tried offline first, and the
    network is only involved when there is genuinely nothing to load.
    """
    global _model
    if _model is not None:
        return _model

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _model = SentenceTransformer(name, device=device, local_files_only=True)
    except Exception:
        # Not cached yet - this is the download, and it is meant to take a while.
        _model = SentenceTransformer(name, device=device)
    return _model


# English averages roughly 1.3 sub-word tokens per word; leave headroom so a
# chunk of technical text doesn't quietly cross the limit.
TOKENS_PER_WORD = 1.45


def max_input_words() -> int:
    """How many words the embedding model can actually see in one chunk.

    Anything past this is silently truncated - the text still reaches the
    answering model, but contributes nothing to whether the chunk is found.
    """
    return int(get_model().max_seq_length / TOKENS_PER_WORD)


def embed(texts: list[str], batch_size: int = 32, progress: bool = False) -> np.ndarray:
    """Embed a list of strings into a (len(texts), dim) float32 array.

    Vectors are unit-normalised, so a dot product between two of them is
    exactly their cosine similarity.
    """
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype="float32")

    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=progress,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")
