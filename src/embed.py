# embed.py
from sentence_transformers import SentenceTransformer
from typing import List

# Load model once at import time
# all-mpnet-base-v2 → 768-dimensional embeddings
_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def embed_text(text: str) -> List[float]:
    """
    Embed a piece of text using the all-mpnet-base-v2 local model.
    Returns a Python list of floats (768-dimensional vector).
    """
    if not isinstance(text, str):
        raise ValueError("embed_text() expects a string")

    vector = _model.encode(text, convert_to_numpy=True)
    return vector.tolist()
