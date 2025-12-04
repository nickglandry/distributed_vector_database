# compute_server.py
import hashlib
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

NUM_SHARDS = 2
STORAGE_NODES = {
    0: "http://localhost:8001",
    1: "http://localhost:8002"
}

app = FastAPI(title="Compute Server")


def compute_shard(vector_id: str) -> int:
    h = hashlib.sha256(vector_id.encode()).hexdigest()
    return int(h, 16) % NUM_SHARDS


class VectorPayload(BaseModel):
    id: str
    vector: List[float]


class SearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 3


# ============ STORE =================

@app.post("/store")
def store(payload: VectorPayload):
    shard = compute_shard(payload.id)
    url = STORAGE_NODES[shard]

    r = requests.post(f"{url}/store", json=payload.dict())
    if not r.ok:
        raise HTTPException(r.status_code, r.text)
    return r.json()


# ============ GET =================

@app.get("/get/{vector_id}")
def get_vec(vector_id: str):
    shard = compute_shard(vector_id)
    url = STORAGE_NODES[shard]

    r = requests.get(f"{url}/get/{vector_id}")
    if not r.ok:
        raise HTTPException(r.status_code, r.text)
    return r.json()


# ============ SEARCH (brute-force) =================

def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0
    return dot / (na * nb)


@app.post("/search")
def search(req: SearchRequest):
    query = req.query_vector
    results = []

    # Loop through 2 shards
    for shard_id, url in STORAGE_NODES.items():
        ids_resp = requests.get(f"{url}/list_ids").json()
        ids = ids_resp.get("ids", [])

        for vid in ids:
            vec_resp = requests.get(f"{url}/get/{vid}")
            if not vec_resp.ok:
                continue
            
            vec_data = vec_resp.json()
            score = cosine(query, vec_data["vector"])

            results.append({
                "id": vid,
                "score": float(score),
                "shard": shard_id
            })

    # Sort by similarity
    results.sort(key=lambda x: x["score"], reverse=True)

    return {"results": results[:req.top_k]}


# Lets the test code know that the server is up and running
@app.get("/")
def root():
    return {"status": "ok"}