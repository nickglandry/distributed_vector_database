# server_launcher.py
import subprocess
import os
import time
import signal
import sys
from dotenv import load_dotenv

load_dotenv()

EMBED_DIM = int(os.getenv('EMBED_DIM'))
NUM_SHARDS = int(os.getenv('NUM_SHARDS'))
processes = []

def start_storage_server(shard_id, port):
    """Start one storage server with SHARD_ID env variable."""
    env = os.environ.copy()
    env["SHARD_ID"] = str(shard_id)

    cmd = [
        sys.executable,       # uses venv python
        "-m", "uvicorn",      # runs uvicorn module
        "storage_server:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]

    print(f"Starting storage server shard {shard_id} on port {port}...")
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)


def start_compute_server(port=9000):
    """Start compute server."""
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "compute_server:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]


    print(f"Starting compute server on port {port}...")
    p = subprocess.Popen(cmd)
    processes.append(p)


def cleanup():
    """Kill all running servers."""
    print("\nShutting down servers...")
    for p in processes:
        try:
            p.terminate()
        except:
            pass

    time.sleep(1)

    for p in processes:
        try:
            p.kill()
        except:
            pass

    print("All servers stopped.")
    sys.exit(0)


def main():
    print("=== Starting Simple 2-Shard Vector DB ===\n")

    for i in range(0, NUM_SHARDS):
        port = f'800{i+1}'
        port = int(port)
        start_storage_server(shard_id=i, port=port)
        time.sleep(0.3)

    # Start compute server
    start_compute_server(port=9000)

    print("\n=============================================")
    print("🚀 Servers are running!") # clearly vibecoded emoji use
    print("Press CTRL+C to stop everything.\n")
    print("=============================================")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
