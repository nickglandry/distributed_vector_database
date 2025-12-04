# compute_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import requests

NUM_SHARDS = 2

STORAGE_NODES: Dict[int, str] = {
    0: "http://localhost:8001",
    1: "http://localhost:8002",
    2: "http://localhost:8003",
    3: "http://localhost:8004",
    4: "http://localhost:8005",
    5: "http://localhost:8006",
    6: "http://localhost:8007",
    7: "http://localhost:8008",
}

# Centroids must match the dimensionality of your vectors
CENTROIDS: Dict[int, List[float]] = {
    0: [0.0, 0.0, 0.0],
    1: [0.0, 0.0, 1.0],
    2: [0.0, 1.0, 0.0],
    3: [1.0, 0.0, 0.0],
    4: [0.0, 1.0, 1.0],
    5: [1.0, 0.0, 1.0],
    6: [1.0, 1.0, 0.0],
    7: [1.0, 1.0, 1.0],
}


def euclidean(a, b):
    return float(np.linalg.norm(a - b))


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def nearest_shards(vec: List[float], m=1):
    v = np.array(vec, dtype=float)
    dists = []
    for shard_id, centroid in CENTROIDS.items():
        c = np.array(centroid, dtype=float)
        d = euclidean(v, c)
        dists.append((d, shard_id))
    dists.sort(key=lambda x: x[0])
    return [sid for (_, sid) in dists[:m]]


app = FastAPI(title="Compute Server")


class VectorPayload(BaseModel):
    id: str
    vector: List[float]


class SearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 5
    shards_to_search: int = 1


@app.get("/")
def root():
    return {
        "centroids": CENTROIDS,
        "nodes": STORAGE_NODES,
    }


@app.post("/store")
def store(payload: VectorPayload):
    shard = nearest_shards(payload.vector, m=1)[0]
    url = STORAGE_NODES[shard]

    r = requests.post(f"{url}/store", json=payload.dict())
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return {"stored_in": shard, "response": r.json()}


@app.get("/get/{vector_id}")
def get_vec(vector_id: str):
    # naive: try all shards
    for shard_id, url in STORAGE_NODES.items():
        r = requests.get(f"{url}/get/{vector_id}")
        if r.status_code == 200:
            d = r.json()
            d["found_in"] = shard_id
            return d
    raise HTTPException(status_code=404, detail="ID not found in any shard")


@app.post("/search")
def search(req: SearchRequest):
    vec = np.array(req.query_vector, dtype=float)
    target_shards = nearest_shards(req.query_vector, m=req.shards_to_search)

    candidates = []

    for shard_id in target_shards:
        url = STORAGE_NODES[shard_id]

        # get all IDs stored in that shard
        ids_resp = requests.get(f"{url}/list_ids")
        ids = ids_resp.json().get("ids", [])

        for vid in ids:
            vec_resp = requests.get(f"{url}/get/{vid}")
            if not vec_resp.ok:
                continue

            item = vec_resp.json()
            v = np.array(item["vector"], dtype=float)
            score = cosine(vec, v)

            candidates.append({
                "id": vid,
                "score": score,
                "shard": shard_id
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return {"results": candidates[:req.top_k]}
