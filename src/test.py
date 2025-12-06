# test.py
import time
import requests
import hashlib
import json
import numpy as np
import os
from embed import embed_text
from datasets import load_dataset
from dotenv import load_dotenv
from cluster import compute_centroids

load_dotenv()

# ==============================
# Setup
# ==============================
EMBED_DIM = int(os.getenv('EMBED_DIM'))
NUM_SHARDS = int(os.getenv('NUM_SHARDS'))
COMPUTE   = "http://localhost:9000"
STORAGE_NODES = [
    f"http://localhost:800{int(node+1)}"
    for node in range(0, NUM_SHARDS)
]

def load_data():
    ds = load_dataset("ag_news", split="train[:500]")

    vectors = []
    for item in ds:
        vectors.append({
            "id": item["text"],
            "vector": embed_text(item["text"])
        })

    return vectors


def find_centroids(vectors):
    np.random.shuffle(vectors)
    sample_vectors = vectors[: int(len(vectors) / 10)] # Use random 10% of initial dataset to get sample vectors
    centroids = compute_centroids(sample_vectors, NUM_SHARDS)
    centroid_map = {i: c for i, c in enumerate(centroids)}
    r = requests.post(f"{COMPUTE}/set_centroids", json=centroid_map)
    if not r.ok:
        print(f"Error setting centroids: {r.status_code}")
    else:
        print(f"Centroids set successfully as {centroids}")


# ==============================
# Helpers
# ==============================

def pretty(x):
    return json.dumps(x, indent=4)


def wait_for_servers():
    print("Waiting for servers to start...")
    for url in STORAGE_NODES + [COMPUTE]:
        ready = False
        while not ready:
            try:
                r = requests.get(url + "/")
                ready = r.ok
            except:
                pass
            if not ready:
                print(f"  Waiting on {url}...")
                time.sleep(0.4)
        print(f"  {url} is up!")
    print("All servers ready.\n")


# ==============================
# Tests
# ==============================

def test_store_vectors():
    print("=== TEST: STORE vectors via compute_server ===")

    # str1 = "The quick brown fox jumped over the lazy dog."
    # str2 = "The Eiffel Tower is in Paris."
    # str3 = "Artificial intelligence is transforming the way people work and learn."
    # str4 = "Data scientists often rely on vector embeddings to capture semantic meaning from text."

    vectors = load_data()

    vector_values = [
        val["vector"]
        for val in vectors
    ]
    find_centroids(vector_values)

    for v in vectors:
        requests.post(f"{COMPUTE}/store", json=v)
    
    print(f"Successfully loaded in {len(vectors)} vectors \n")


def test_list_shards():
    print("=== TEST: LIST IDS from each shard ===")

    for i, url in enumerate(STORAGE_NODES):
        try:
            r = requests.get(f"{url}/list_ids")
            print(f"Shard {i} loaded successfully")
        except Exception as e:
            print(f"Shard {i} error:", e)
    print()

def test_search():
    print("=== TEST: SEARCH via compute server ===")

    query_str  = "Historical wars in Europe."
    r = requests.post(
        f"{COMPUTE}/search",
        json={"query_vector": embed_text(query_str), "top_k": 3}
    )

    print("Search results:")
    print(pretty(r.json()))
    print()


# ==============================
# Main
# ==============================

def main():
    wait_for_servers()
    test_store_vectors()
    test_list_shards()
    test_search()
    print("=== Tests completed ===")


if __name__ == "__main__":
    main()
