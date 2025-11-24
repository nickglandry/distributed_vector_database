from fastapi import FastAPI
import numpy as np
import requests
import faiss

app = FastAPI()

index = None
dims = None

@app.post("/init_index")
def init_index():
    global index, dims

    # Ask storage node for vector shape
    info = requests.get("http://localhost:8000/get_vectors").json()
    count, dims = info["count"], info["dims"]

    # Fetch all vectors
    response = requests.post("http://localhost:8000/fetch_partition", json={"start": 0, "end": count})
    vectors = np.array(response.json(), dtype="float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(dims)
    index.add(vectors)

    return {"status": "index built", "vector_count": count}

@app.post("/query")
def query(k: int = 5):
    # Generate a random query vector
    query_vec = np.random.random((1, dims)).astype("float32")

    distances, indices = index.search(query_vec, k)

    return {
        "query_vector": query_vec.tolist(),
        "neighbors": indices.tolist(),
        "distances": distances.tolist()
    }
