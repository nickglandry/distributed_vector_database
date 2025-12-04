# storage_server.py
import os
import json
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
NUM_SHARDS = 2

# File for this shard
SHARD_FILE = f"data/shard_{SHARD_ID}.json"

# Ensure file exists
if not os.path.exists(SHARD_FILE):
    with open(SHARD_FILE, "w") as f:
        json.dump({}, f)

app = FastAPI(title=f"Storage Server Shard {SHARD_ID}")


def compute_shard(vector_id: str) -> int:
    h = hashlib.sha256(vector_id.encode()).hexdigest()
    return int(h, 16) % NUM_SHARDS


def read_shard() -> Dict[str, List[float]]:
    """Always read the shard file from disk."""
    with open(SHARD_FILE, "r") as f:
        return json.load(f)


def write_shard(data: Dict[str, List[float]]):
    """Always write data back to the shard file."""
    with open(SHARD_FILE, "w") as f:
        json.dump(data, f)


class VectorPayload(BaseModel):
    id: str
    vector: List[float]


@app.get("/")
def root():
    return {"status": "ok", "shard": SHARD_ID}


@app.post("/store")
def store_vec(payload: VectorPayload):
    shard = compute_shard(payload.id)
    if shard != SHARD_ID:
        raise HTTPException(
            status_code=400,
            detail=f"ID {payload.id} belongs to shard {shard}, not shard {SHARD_ID}",
        )

    db = read_shard()
    db[payload.id] = payload.vector
    write_shard(db)

    return {"status": "stored", "id": payload.id, "shard": SHARD_ID}


@app.get("/get/{vector_id}")
def get_vec(vector_id: str):
    shard = compute_shard(vector_id)
    if shard != SHARD_ID:
        raise HTTPException(
            status_code=400,
            detail=f"ID {vector_id} belongs to shard {shard}, not this shard",
        )

    db = read_shard()
    if vector_id not in db:
        raise HTTPException(404, "Vector not found")

    return {"id": vector_id, "vector": db[vector_id], "shard": SHARD_ID}


@app.get("/list_ids")
def list_ids():
    db = read_shard()
    return {"count": len(db), "ids": list(db.keys()), "shard": SHARD_ID}
