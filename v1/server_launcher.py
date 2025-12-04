# server_launcher.py
import subprocess
import os
import time
import signal
import sys

processes = []


def start_storage_server(shard_id, port):
    """Start one storage server with SHARD_ID env variable."""
    env = os.environ.copy()
    env["SHARD_ID"] = str(shard_id)

    cmd = [
        "uvicorn",
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
        "uvicorn",
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

    # Start shard 0
    start_storage_server(shard_id=0, port=8001)
    time.sleep(0.3)

    # Start shard 1
    start_storage_server(shard_id=1, port=8002)
    time.sleep(0.3)

    # Start compute server
    start_compute_server(port=9000)

    print("\n=============================================")
    print("🚀 Servers are running!")
    print("Storage shard 0 → http://localhost:8001")
    print("Storage shard 1 → http://localhost:8002")
    print("Compute server → http://localhost:9000")
    print("=============================================")
    print("Press CTRL+C to stop everything.\n")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
