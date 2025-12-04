# test.py
import time
import requests
import hashlib
import json
import numpy as np

# ==============================
# Config
# ==============================

STORAGE_0 = "http://localhost:8001"
STORAGE_1 = "http://localhost:8002"
COMPUTE   = "http://localhost:9000"

STORAGE_NODES = [STORAGE_0, STORAGE_1]


# ==============================
# Helpers
# ==============================

def pretty(x):
    return json.dumps(x, indent=4)


def compute_shard(id_str):
    """Must match storage_server + compute_server logic."""
    h = hashlib.sha256(id_str.encode()).hexdigest()
    return int(h, 16) % 2


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

    sample_vectors = [
        {"id": "vecA", "vector": [0.1, 0.2, 0.3]},
        {"id": "vecB", "vector": [0.0, 0.5, 0.9]},
        {"id": "vecC", "vector": [0.9, 0.1, 0.2]},
        {"id": "vecD", "vector": [0.2, 0.8, 0.4]},
    ]

    for v in sample_vectors:
        shard = compute_shard(v["id"])
        print(f"Storing {v['id']} → shard {shard}")
        resp = requests.post(f"{COMPUTE}/store", json=v)
        print("Response:", pretty(resp.json()))
    print()


def test_list_shards():
    print("=== TEST: LIST IDS from each shard ===")

    for i, url in enumerate(STORAGE_NODES):
        try:
            r = requests.get(f"{url}/list_ids")
            print(f"Shard {i}:", pretty(r.json()))
        except Exception as e:
            print(f"Shard {i} error:", e)
    print()


def test_get_vectors():
    print("=== TEST: GET vectors individually ===")

    for vid in ["vecA", "vecB", "vecC", "vecD"]:
        shard = compute_shard(vid)
        url = STORAGE_NODES[shard]
        print(f"GET {vid} (should be on shard {shard})")
        r = requests.get(f"{url}/get/{vid}")
        print(pretty(r.json()))
    print()


def test_search():
    print("=== TEST: SEARCH via compute server ===")

    query_vec = [0.0, 0.2, 1.0]
    r = requests.post(
        f"{COMPUTE}/search",
        json={"query_vector": query_vec, "top_k": 3}
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
    test_get_vectors()
    test_search()
    print("=== Tests completed ===")


if __name__ == "__main__":
    main()
