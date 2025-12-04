# NOTE: Run with uvicorn compute_server:app --host 0.0.0.0 --port 8001 --reload

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import requests
import faiss

import embed

app = FastAPI()

index = None
dims = None
STORAGE_URL = "http://localhost:8000"


class TextRequest(BaseModel):
    text: str


class QueryTextRequest(BaseModel):
    text: str
    k: int = 5


# v1/compute_server.py
CHUNK_SIZE = 10_000  # tune based on payload size/memory

@app.post("/init_index")
def init_index():
    global index, dims

    try:
        stats = requests.get(f"{STORAGE_URL}/stats").json()
    except Exception as exc:
        return {"error": "failed to contact storage node", "details": str(exc)}

    if "error" in stats:
        return stats
    count, dims = stats["count"], stats["dims"]
    if dims is None:
        return {"error": "no vectors found and store not initialized"}

    index = faiss.IndexFlatL2(dims)

    # Pull partitions sequentially to avoid huge responses
    added = 0
    for start in range(0, count, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, count)
        try:
            resp = requests.post(
                f"{STORAGE_URL}/fetch_partition", json={"start": start, "end": end}
            )
            resp.raise_for_status()
            block = resp.json()
        except Exception as exc:
            return {"error": "failed to fetch vectors from storage", "details": str(exc), "range": [start, end]}

        if isinstance(block, dict) and "error" in block:
            return block

        if block:
            vecs = np.array(block, dtype="float32")
            index.add(vecs)
            added += vecs.shape[0]

    return {"status": "index built", "vector_count": added, "dims": dims, "chunk_size": CHUNK_SIZE}


def _ensure_index(target_dims: int):
    """Create an empty index if one does not already exist."""
    global index, dims
    if index is None:
        index = faiss.IndexFlatL2(target_dims)
        dims = target_dims


@app.post("/insert_text")
def insert_text(body: TextRequest):
    """Embed text, persist the vector in storage, and add it to the FAISS index."""
    global dims

    try:
        vector = embed.embed_text(body.text)
    except RuntimeError as exc:
        return {"error": str(exc)}

    vector_list = vector.tolist()

    # Ensure local index exists with the correct dimensionality
    if dims is None:
        dims = vector.shape[0]
    if dims != vector.shape[0]:
        return {"error": f"dimension mismatch: index dims {dims}, embedding dims {vector.shape[0]}"}

    _ensure_index(dims)

    # Persist to storage
    try:
        resp = requests.post(f"{STORAGE_URL}/insert_vector", json={"vector": vector_list})
        payload = resp.json()
    except Exception as exc:
        return {"error": "storage insert failed", "details": str(exc)}

    if resp.status_code != 200 or (isinstance(payload, dict) and "error" in payload):
        return {"error": "storage insert failed", "details": payload}

    # Update FAISS index
    index.add(vector.reshape(1, -1))

    return {"status": "inserted", "index_size": index.ntotal, "dims": dims}

@app.post("/query")
def query(k: int = 5):
    if index is None or dims is None:
        return {"error": "index not initialized"}

    # Generate a random query vector
    query_vec = np.random.random((1, dims)).astype("float32")

    distances, indices = index.search(query_vec, k)

    return {
        "query_vector": query_vec.tolist(),
        "neighbors": indices.tolist(),
        "distances": distances.tolist()
    }


@app.post("/query_text")
def query_text(body: QueryTextRequest):
    if index is None or dims is None:
        return {"error": "index not initialized"}

    try:
        vector = embed.embed_text(body.text).reshape(1, -1)
    except RuntimeError as exc:
        return {"error": str(exc)}

    if vector.shape[1] != dims:
        return {"error": f"dimension mismatch: index dims {dims}, embedding dims {vector.shape[1]}"}

    distances, indices = index.search(vector, body.k)

    return {
        "query_text": body.text,
        "neighbors": indices.tolist(),
        "distances": distances.tolist()
    }

@app.post("/add_vector")
def add_vector(vector: list[float]):
    global index, dims
    vec = np.array(vector, dtype="float32").reshape(1, -1)

    if dims is None:
        dims = vec.shape[1]
    if vec.shape[1] != dims:
        return {"error": f"expected dims {dims}"}

    _ensure_index(dims)

    # persist to storage
    try:
        resp = requests.post(f"{STORAGE_URL}/insert_vector", json={"vector": vector})
        payload = resp.json()
    except Exception as exc:
        return {"error": "storage insert failed", "details": str(exc)}

    if resp.status_code != 200:
        return {"error": "storage insert failed", "details": payload}

    index.add(vec)
    return {"status": "inserted", "index_size": index.ntotal}
