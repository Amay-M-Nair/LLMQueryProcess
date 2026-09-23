"""Turn text into vectors.

An embedding is a list of ~384 numbers representing a piece of text's meaning.
Two texts about the same thing end up with vectors pointing in a similar
direction, which is what makes search-by-meaning possible.

LangChain's HuggingFaceEmbeddings does the work. It wraps the same
sentence-transformers model and returns bit-for-bit identical vectors, so
this is a change of plumbing and not of behaviour - checked rather than
assumed, because "it is the same model underneath" is exactly the sort of
thing that turns out not to be.

The import is deferred: it pulls in torch, which takes ~20s, and the web app
should start instantly and pay that on the first embed instead.
"""

from typing import TYPE_CHECKING

import numpy as np

from utils import config

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

# English averages roughly 1.3 sub-word tokens per word; leave headroom so a
# chunk of technical text doesn't quietly cross the limit.
TOKENS_PER_WORD = 1.45

_model = None


def get_model(name: str = config.EMBED_MODEL) -> "HuggingFaceEmbeddings":
    """Load the embedding model once and reuse it (it is slow to construct).

    `local_files_only` is not an optimisation. sentence-transformers contacts
    the HuggingFace hub to check for a newer revision even when the weights
    are cached, and a slow hub blocks startup for minutes with nothing
    printed, which looks exactly like the application having frozen.
    """
    global _model
    if _model is None:
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _model = HuggingFaceEmbeddings(
                model_name=name,
                model_kwargs={"device": device, "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            # Not cached yet - this is the download, and it is meant to take
            # a while.
            _model = HuggingFaceEmbeddings(
                model_name=name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _model


def max_input_words() -> int:
    """How many words the embedding model can actually see in one chunk.

    Anything past this is silently truncated - the text still reaches the
    answering model, but contributes nothing to whether the chunk is found.
    """
    return int(get_model()._client.max_seq_length / TOKENS_PER_WORD)


def embed(texts: list[str], batch_size: int = 32, progress: bool = False) -> np.ndarray:
    """Embed a list of strings into a (len(texts), dim) float32 array.

    Vectors are unit-normalised, so a dot product between two of them is
    exactly their cosine similarity - which is what lets the store use
    FAISS's inner-product index and still report a real cosine.
    """
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype="float32")

    vectors = get_model().embed_documents(list(texts))
    return np.asarray(vectors, dtype="float32")
