# storage_server.py
import os
import json
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
DB_FILE = f"data/shard_{SHARD_ID}.sqlite3"

# Ensure SQLite file exists and has the right schema
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    vector_json TEXT NOT NULL
)
""")
conn.commit()
conn.close()

app = FastAPI(title=f"Shard {SHARD_ID} Storage Server")


class VectorPayload(BaseModel):
    id: str
    vector: List[float]


@app.get("/")
def root():
    return {"status": "ok", "shard": SHARD_ID}


@app.post("/store")
def store_vec(payload: VectorPayload):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Store vector as JSON on disk
    cur.execute(
        "REPLACE INTO vectors (id, vector_json) VALUES (?, ?)",
        (payload.id, json.dumps(payload.vector))
    )

    conn.commit()
    conn.close()
    return {"status": "stored", "id": payload.id, "shard": SHARD_ID}


@app.get("/get/{vector_id}")
def get_vec(vector_id: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("SELECT vector_json FROM vectors WHERE id = ?", (vector_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Vector not found on this shard")

    return {
        "id": vector_id,
        "vector": json.loads(row[0]),
        "shard": SHARD_ID
    }


@app.get("/list_ids")
def list_ids():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("SELECT id FROM vectors")
    ids = [row[0] for row in cur.fetchall()]

    conn.close()
    return {"count": len(ids), "ids": ids, "shard": SHARD_ID}
