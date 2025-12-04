# storage_server.py
import os
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

# Which shard this server owns: 0 or 1
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
NUM_SHARDS = 2

# In-memory store (dictionary)
DB: Dict[str, List[float]] = {}

app = FastAPI(title=f"Storage Server Shard {SHARD_ID}")


def compute_shard(vector_id: str) -> int:
    """Hash-based sharding: always compute the SAME shard."""
    h = hashlib.sha256(vector_id.encode()).hexdigest()
    return int(h, 16) % NUM_SHARDS


class VectorPayload(BaseModel):
    id: str
    vector: List[float]


@app.post("/store")
def store_vec(payload: VectorPayload):
    shard = compute_shard(payload.id)
    if shard != SHARD_ID:
        raise HTTPException(
            status_code=400,
            detail=f"ID {payload.id} belongs to shard {shard}, not shard {SHARD_ID}"
        )

    DB[payload.id] = payload.vector
    return {"status": "stored", "id": payload.id, "shard": SHARD_ID}


@app.get("/get/{vector_id}")
def get_vec(vector_id: str):
    shard = compute_shard(vector_id)
    if shard != SHARD_ID:
        raise HTTPException(
            status_code=400,
            detail=f"ID {vector_id} belongs to shard {shard}, not this shard"
        )

    if vector_id not in DB:
        raise HTTPException(404, "Vector not found")

    return {"id": vector_id, "vector": DB[vector_id], "shard": SHARD_ID}


@app.get("/list_ids")
def all_ids():
    return {"count": len(DB), "ids": list(DB.keys()), "shard": SHARD_ID}


# Lets the test code know that the server is up and running
@app.get("/")
def root():
    return {"status": "ok", "shard": SHARD_ID}