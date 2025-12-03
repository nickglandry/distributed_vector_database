import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "vectors.npy")

vectors = None


class PartitionRequest(BaseModel):
    start: int
    end: int


class InsertRequest(BaseModel):
    vector: list[float]


def _load_vectors_from_disk():
    global vectors
    if vectors is not None:
        return
    if os.path.exists(DATA_PATH):
        vectors = np.load(DATA_PATH)


def _persist():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    np.save(DATA_PATH, vectors)


@app.post("/init_store")
def init_store(dims: int = 128):
    """Initialize an empty store on disk with the given dimension."""
    global vectors
    vectors = np.empty((0, dims), dtype="float32")
    _persist()
    return {"status": "initialized", "count": 0, "dims": dims}


@app.get("/stats")
def stats():
    _load_vectors_from_disk()
    if vectors is None:
        return {"count": 0, "dims": None}
    return {"count": vectors.shape[0], "dims": vectors.shape[1]}


@app.post("/insert_vector")
def insert_vector(body: InsertRequest):
    global vectors
    _load_vectors_from_disk()

    vec = np.array(body.vector, dtype="float32")

    if vectors is None:
        # Initialize based on first vector
        vectors = np.empty((0, vec.shape[0]), dtype="float32")

    if vec.shape != (vectors.shape[1],):
        return {"error": f"dimension mismatch: expected {vectors.shape[1]}"}

    vectors = np.vstack([vectors, vec])
    _persist()

    return {"status": "inserted", "count": vectors.shape[0], "dims": vectors.shape[1]}


@app.post("/fetch_partition")
def fetch_partition(range: PartitionRequest):
    _load_vectors_from_disk()
    if vectors is None:
        return {"error": "no vectors found"}

    start, end = range.start, range.end

    if start < 0 or end < 0 or start > end or end > vectors.shape[0]:
        return {"error": "invalid range", "count": vectors.shape[0]}

    return vectors[start:end].tolist()
