# Distributed Vector Database

A minimal end-to-end vector database demo: local embedding, sharding, storage, and retrieval over FastAPI. Each shard stores vectors in SQLite; a compute node routes writes/reads/searches across shards using centroids.

## What’s here
- src/embed.py — sentence-transformers embedder (all-mpnet-base-v2, 768 dims).
- src/compute_server.py — FastAPI router that assigns shards, forwards storage/get, and runs cosine search across selected shards.
- src/storage_server.py — FastAPI shard storing vectors in SQLite (`src/data/shard_*.sqlite3`), exposes store/get/list_ids.
- src/cluster.py — KMeans centroid computation to pick shard centroids.
- src/server_launcher.py — spins up N storage shards + the compute server via uvicorn.
- src/test.py — simple load/benchmark using the AG News dataset, populates shards, runs search multiple times.
- report/benchmarks.py — benchmark results and plotting helpers for varying embedding sizes and shard counts.

## Requirements
Python 3.10+ recommended. Install dependencies:


## Configuration
Set these env vars (see src/.env):

## Run the cluster
From the `src` directory:

- Storage shards start on ports 8001..(8000+NUM_SHARDS).
- Compute server starts on port 9000.
- Use Ctrl+C to stop (launcher cleans up child processes).

## API overview (compute server on :9000)
- `GET /` — returns centroids and shard URLs.
- `POST /set_centroids` — body: `{shard_id: [centroid_vec], ...}` to override centroids.
- `POST /store` — body: `{"id": "string", "vector": [floats]}`; routes to nearest shard.
- `GET /get/{id}` — fetch by id (naively checks all shards).
- `POST /search` — body: `{"query_vector": [...], "top_k": 5, "shards_to_search": 1}`; cosine search over nearest shards.

## Storage shard API (each shard on :8001+id)
- `GET /` — health.
- `POST /store` — same payload as above; stores in SQLite.
- `GET /get/{id}` — fetch vector from that shard.
- `GET /list_ids` — list all ids on that shard.

## Example workflow
1) Start servers (`python server_launcher.py`).
2) In another shell, run tests/loader:

`cd src`          
`python test.py`

- Loads 10k AG News samples, embeds, computes centroids (10% sample), sets centroids, stores all vectors, then runs repeated searches and prints timings.

## Benchmarking
- `report/benchmarks.py` holds recorded timings for various embedding sizes (`e{dim}`) and shard counts (`s{shards}`) plus plotting helpers. Adjust figures as needed.

## Notes
- Ensure `EMBED_DIM` matches the embedder or centroid math will misalign.
- Data persists under `src/data/shard_*.sqlite3`; delete those files to reset.
- Centroids are random at startup unless you call `/set_centroids` (test.py does this).

## Disclosure of Generative AI Use
Portions of this repository’s documentation and code were drafted with the assistance of Large Language Models. This project is intended to be a prototype, meaning the overall quality of code was not strongly considered. Users should review code for accuracy, security, and correctness before use in any critical applicaitons.