from fastapi import FastAPI
import numpy as np

app = FastAPI()

# In-memory vector store for prototype
vectors = None

@app.post("/load_vectors")
def load_vectors(count: int = 10000, dims: int = 128):
    global vectors
    vectors = np.random.random((count, dims)).astype("float32")
    return {"status": "loaded", "count": count, "dims": dims}

@app.get("/get_vectors")
def get_vectors():
    return {"count": vectors.shape[0], "dims": vectors.shape[1]}

@app.post("/fetch_partition")
def fetch_partition(start: int, end: int):
    return vectors[start:end].tolist()
