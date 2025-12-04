"""Quick smoke test to insert and query a sentence via the running servers."""

import requests


STORAGE_URL = "http://localhost:8000"
COMPUTE_URL = "http://localhost:8001"


def ensure_store(dims: int = 384):
    """Initialize storage if it is empty; otherwise return existing stats."""
    stats_resp = requests.get(f"{STORAGE_URL}/stats")
    stats_resp.raise_for_status()
    stats = stats_resp.json()

    if stats.get("dims") is None:
        init_resp = requests.post(f"{STORAGE_URL}/init_store", params={"dims": dims})
        init_resp.raise_for_status()
        return init_resp.json()

    return stats


def init_index():
    resp = requests.post(f"{COMPUTE_URL}/init_index")
    resp.raise_for_status()
    return resp.json()


def insert_sentence(text: str):
    resp = requests.post(f"{COMPUTE_URL}/insert_text", json={"text": text})
    resp.raise_for_status()
    return resp.json()


def query_sentence(text: str, k: int = 3):
    resp = requests.post(f"{COMPUTE_URL}/query_text", json={"text": text, "k": k})
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    print("storage:", ensure_store())
    print("init_index:", init_index())
    print("insert:", insert_sentence("hello world"))
    print("insert:", insert_sentence("Welcome to New York"))
    print("query:", query_sentence("hello world"))
