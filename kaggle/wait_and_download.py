"""Wait for a running Kaggle kernel to finish and download its outputs.

Usage:
    python kaggle/wait_and_download.py <kernel_name>
"""

import os
import sys
import time
import json
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def run_cmd(cmd_list: list, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[CMD] {' '.join(cmd_list)}", flush=True)
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[ERROR] Stderr: {res.stderr}", flush=True)
        print(f"[ERROR] Stdout: {res.stdout}", flush=True)
        res.check_returncode()
    return res


def wait_and_download(kernel_name: str, poll_interval_sec: int = 20, timeout_sec: int = 32400):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_dir = os.path.join(repo_root, "kaggle", "kernels", kernel_name)
    output_dir = os.path.join(repo_root, "results", "raw", kernel_name)
    os.makedirs(output_dir, exist_ok=True)

    meta_file = os.path.join(kernel_dir, "kernel-metadata.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    kernel_slug = meta["id"]

    print(f"Waiting for kernel {kernel_slug} to complete (polling every {poll_interval_sec}s)...", flush=True)
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            print(f"[TIMEOUT] Exceeded timeout of {timeout_sec}s", flush=True)
            sys.exit(1)

        status_res = run_cmd(["kaggle", "kernels", "status", kernel_slug], check=False)
        stdout = status_res.stdout.strip()
        print(f"[{time.strftime('%H:%M:%S')}] {stdout}", flush=True)

        if "complete" in stdout.lower():
            print(f"Kernel {kernel_slug} completed successfully!", flush=True)
            break
        elif "error" in stdout.lower() or "fail" in stdout.lower() or "cancel" in stdout.lower():
            print(f"Kernel {kernel_slug} failed with status: {stdout}", flush=True)
            run_cmd(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir], check=False)
            sys.exit(1)

        time.sleep(poll_interval_sec)

    print(f"Downloading outputs to {output_dir}...", flush=True)
    run_cmd(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir])
    print(f"Outputs successfully downloaded to {output_dir}", flush=True)
    print("Files downloaded:")
    for f in os.listdir(output_dir):
        sz = os.path.getsize(os.path.join(output_dir, f))
        print(f"  - {f} ({sz} bytes)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wait_and_download.py <kernel_name>")
        sys.exit(1)
    wait_and_download(sys.argv[1])
