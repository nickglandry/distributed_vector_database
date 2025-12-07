# storage_server.py
import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

# Milvus Lite persists to a local file
os.makedirs("data", exist_ok=True)
MILVUS_URI = os.path.join("data", f"milvus_shard_{SHARD_ID}.db")
COLLECTION_NAME = f"shard_{SHARD_ID}_vectors"

connections.connect(alias="default", uri=MILVUS_URI)

if utility.has_collection(COLLECTION_NAME):
    collection = Collection(COLLECTION_NAME)
else:
    id_field = FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        is_primary=True,
        max_length=256,
    )
    vector_field = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=EMBED_DIM,
    )
    schema = CollectionSchema(
        fields=[id_field, vector_field],
        description=f"Vectors for shard {SHARD_ID}",
    )
    collection = Collection(name=COLLECTION_NAME, schema=schema)

if not collection.indexes:
    collection.create_index(
        field_name="vector",
        index_params={"index_type": "FLAT", "metric_type": "L2", "params": {}},
    )
collection.load()

app = FastAPI(title=f"Shard {SHARD_ID} Storage Server")


class VectorPayload(BaseModel):
    id: str
    vector: List[float]


class SearchPayload(BaseModel):
    query_vector: List[float]
    top_k: int = 5
    metric: str = "L2"  # IP (inner product) or L2


@app.get("/")
def root():
    return {"status": "ok", "shard": SHARD_ID}


@app.post("/store")
def store_vec(payload: VectorPayload):
    if len(payload.vector) != EMBED_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dim {len(payload.vector)} != EMBED_DIM {EMBED_DIM}",
        )

    # Columnar insert: [id_column, vector_column]
    collection.insert([[payload.id], [payload.vector]])
    collection.flush()
    if not collection.has_index():
        collection.create_index(
            field_name="vector",
            index_params={"index_type": "FLAT", "metric_type": "L2", "params": {}},
        )
    collection.load()

    return {"status": "stored", "id": payload.id, "shard": SHARD_ID}


@app.get("/get/{vector_id}")
def get_vec(vector_id: str):
    rows = collection.query(
        expr=f'id == "{vector_id}"',
        output_fields=["id", "vector"],
        limit=1,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Vector not found on this shard")

    row = rows[0]
    return {"id": row["id"], "vector": row["vector"], "shard": SHARD_ID}


@app.post("/search")
def search(req: SearchPayload):
    if len(req.query_vector) != EMBED_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Query dim {len(req.query_vector)} != EMBED_DIM {EMBED_DIM}",
        )

    search_params = {"metric_type": req.metric, "params": {"nprobe": 10}}
    if not collection.indexes:
        collection.create_index(
            field_name="vector",
            index_params={"index_type": "FLAT", "metric_type": "L2", "params": {}},
        )
    collection.load()
    results = collection.search(
        data=[req.query_vector],
        anns_field="vector",
        param=search_params,
        limit=req.top_k,
        output_fields=["id", "vector"],
    )

    hits = [
        {
            "id": hit.entity.get("id"),
            "vector": hit.entity.get("vector"),
            "score": float(hit.score),
            "shard": SHARD_ID,
        }
        for hit in results[0]
    ]
    return {"results": hits}


@app.get("/list_ids")
def list_ids():
    rows = collection.query(expr='id != ""', output_fields=["id"])
    ids = [row["id"] for row in rows]
    return {"count": len(ids), "ids": ids, "shard": SHARD_ID}
