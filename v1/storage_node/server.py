import os
from fastapi import FastAPI
import numpy as np

app = FastAPI()

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "vectors.npy")

vectors = None

@app.post("/load_vectors")
def load_vectors(count: int = 10000, dims: int = 128):
    global vectors
    vectors = np.random.random((count, dims)).astype("float32")

    # Save to disk
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    np.save(DATA_PATH, vectors)

    return {"status": "loaded and saved", "count": count, "dims": dims}

@app.post("/load_from_disk")
def load_from_disk():
    global vectors

    if not os.path.exists(DATA_PATH):
        return {"error": "no saved vectors found"}

    vectors = np.load(DATA_PATH)

    return {"status": "loaded from disk", "count": vectors.shape[0], "dims": vectors.shape[1]}

@app.get("/get_vectors")
def get_vectors():
    return {"count": vectors.shape[0], "dims": vectors.shape[1]}

@app.post("/fetch_partition")
def fetch_partition(start: int, end: int):
    return vectors[start:end].tolist()
