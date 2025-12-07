# compute_server.py
import os
from typing import Dict, List

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import quote

load_dotenv()

EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
NUM_SHARDS = int(os.getenv("NUM_SHARDS", "1"))
METRIC_DEFAULT = os.getenv("METRIC", "L2")  # IP or L2

STORAGE_NODES: Dict[int, str] = {
    shard: f"http://localhost:800{shard + 1}"
    for shard in range(NUM_SHARDS)
}

# Centroids can be set by client; default random
CENTROIDS: Dict[int, List[float]] = {
    shard: np.random.randn(EMBED_DIM).tolist()
    for shard in range(NUM_SHARDS)
}


def euclidean(a, b):
    return float(np.linalg.norm(a - b))


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
    metric: str = METRIC_DEFAULT  # IP or L2


@app.post("/set_centroids")
def set_centroids(centroids: Dict[int, List[float]]):
    global CENTROIDS
    CENTROIDS = centroids
    return {"status": "centroids updated", "count": len(centroids)}


@app.get("/")
def root():
    return {"centroids": CENTROIDS, "nodes": STORAGE_NODES}


@app.post("/store")
def store(payload: VectorPayload):
    if len(payload.vector) != EMBED_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dim {len(payload.vector)} != EMBED_DIM {EMBED_DIM}",
        )
    shard = nearest_shards(payload.vector, m=1)[0]
    url = STORAGE_NODES[shard]

    r = requests.post(f"{url}/store", json=payload.dict())
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return {"stored_in": shard, "response": r.json()}


@app.get("/get/{vector_id}")
def get_vec(vector_id: str):
    for shard_id, url in STORAGE_NODES.items():
        encoded = quote(vector_id, safe="")
        r = requests.get(f"{url}/get/{encoded}")
        if r.status_code == 200:
            d = r.json()
            d["found_in"] = shard_id
            return d
    raise HTTPException(status_code=404, detail="ID not found in any shard")

@app.post("/search")
def search(req: SearchRequest):
    if len(req.query_vector) != EMBED_DIM:
        raise HTTPException(status_code=400, detail=f"Query dim {len(req.query_vector)} != EMBED_DIM {EMBED_DIM}")

    target_shards = nearest_shards(req.query_vector, m=req.shards_to_search)
    aggregated = []
    metric = req.metric.upper()  # will be L2 by default

    for shard_id in target_shards:
        url = STORAGE_NODES[shard_id]
        resp = requests.post(
            f"{url}/search",
            json={"query_vector": req.query_vector, "top_k": req.top_k, "metric": metric},
        )
        if not resp.ok:
            continue
        for item in resp.json().get("results", []):
            aggregated.append({"id": item.get("id"), "score": item.get("score"), "shard": shard_id})

    aggregated.sort(key=lambda x: x["score"])  # L2: smaller is better
    return {"results": aggregated[: req.top_k]}
