import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - environment may lack dependency during dev
    SentenceTransformer = None


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

def _get_model():
    """Lazily load the single embedding model we support."""
    global _model

    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it (e.g. pip install sentence-transformers) "
            "and ensure the environment can download the model."
        )

    if _model is None:
        _model = SentenceTransformer(DEFAULT_MODEL)

    return _model


def embed_text(text: str) -> np.ndarray:
    """Encode text into a float32 vector with a fixed model."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return np.array(embedding, dtype="float32")
