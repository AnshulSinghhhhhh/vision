"""Kaggle kernel orchestration: packaging code dataset, pushing kernels, polling status, and fetching results.

Security:
- Never prints, logs, or stores Kaggle credentials.
- Reads credentials exclusively from Kaggle CLI's authenticated environment.
"""

import os
import sys
import time
import json
import shutil
import subprocess
from typing import Dict, Any, Optional

# Ensure prints flush immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

CODE_DATASET_SLUG = "knobs-code"


def run_cmd(cmd_list: list, check: bool = True) -> subprocess.CompletedProcess:
    """Executes a command and returns the completed process."""
    print(f"[CMD] {' '.join(cmd_list)}", flush=True)
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[ERROR] Stderr: {res.stderr}", flush=True)
        print(f"[ERROR] Stdout: {res.stdout}", flush=True)
        res.check_returncode()
    return res


def package_and_push_code_dataset(
    repo_root: str,
    staging_dir: str,
    username: str = "anshulsingh45",
):
    """Stages clean repository code and pushes/updates the private Kaggle code dataset."""
    staging_dir = os.path.abspath(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    
    # Copy src, pyproject.toml, requirements.txt
    src_dest = os.path.join(staging_dir, "src")
    if os.path.exists(src_dest):
        shutil.rmtree(src_dest)
    shutil.copytree(os.path.join(repo_root, "src"), src_dest)
    shutil.copy2(os.path.join(repo_root, "pyproject.toml"), staging_dir)
    shutil.copy2(os.path.join(repo_root, "requirements.txt"), staging_dir)
    
    # Write GIT_COMMIT provenance file
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True)
        commit_hash = res.stdout.strip()
        status_res = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True)
        is_dirty = bool(status_res.stdout.strip())
        git_commit_info = f"{commit_hash}{'-dirty' if is_dirty else ''}"
    except Exception:
        git_commit_info = "unknown_git_commit"

    for c_path in [
        os.path.join(staging_dir, "GIT_COMMIT"),
        os.path.join(src_dest, "GIT_COMMIT"),
        os.path.join(src_dest, "knobs", "GIT_COMMIT"),
    ]:
        os.makedirs(os.path.dirname(c_path), exist_ok=True)
        with open(c_path, "w", encoding="utf-8") as f:
            f.write(git_commit_info)
    print(f"Recorded provenance in GIT_COMMIT: {git_commit_info}")
    
    # Metadata for dataset
    meta_path = os.path.join(staging_dir, "dataset-metadata.json")
    meta = {
        "title": "knobs-code",
        "id": f"{username}/{CODE_DATASET_SLUG}",
        "licenses": [{"name": "apache-2.0"}],
        "is_private": True,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Checking if dataset {username}/{CODE_DATASET_SLUG} exists on Kaggle...")
    check = run_cmd(["kaggle", "datasets", "status", f"{username}/{CODE_DATASET_SLUG}"], check=False)
    
    if check.returncode == 0:
        print("Dataset exists. Creating new version...")
        run_cmd([
            "kaggle", "datasets", "version",
            "-p", staging_dir,
            "-m", f"Update code {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "--dir-mode", "zip"
        ])
    else:
        print("Dataset does not exist. Initializing dataset...")
        run_cmd(["kaggle", "datasets", "create", "-p", staging_dir, "--dir-mode", "zip"])
        
    print("Waiting for dataset to be ready...")
    for _ in range(30):
        st = run_cmd(["kaggle", "datasets", "status", f"{username}/{CODE_DATASET_SLUG}"], check=False)
        if "ready" in st.stdout.lower():
            print("Dataset is ready.")
            break
        time.sleep(3)
    print("Code dataset push complete.")


def push_and_wait_kernel(
    kernel_dir: str,
    kernel_slug: str,
    output_dir: str,
    accelerator: str = "NvidiaTeslaT4",
    poll_interval_sec: int = 60,
    timeout_sec: int = 14400,  # 4 hours
) -> bool:
    """Pushes a kernel, polls its execution until completion, and downloads outputs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Push kernel
    push_cmd = ["kaggle", "kernels", "push", "-p", kernel_dir]
    if accelerator:
        push_cmd.extend(["--accelerator", accelerator])
        
    print(f"Pushing kernel from {kernel_dir} with accelerator {accelerator}...")
    run_cmd(push_cmd)
    
    # 2. Poll status
    start_time = time.time()
    print(f"Polling status for {kernel_slug} every {poll_interval_sec}s...")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            print(f"[TIMEOUT] Kernel {kernel_slug} exceeded timeout of {timeout_sec}s")
            return False
            
        status_res = run_cmd(["kaggle", "kernels", "status", kernel_slug], check=False)
        stdout = status_res.stdout.strip()
        print(f"[{time.strftime('%H:%M:%S')}] {stdout}")
        
        if "complete" in stdout.lower():
            print(f"Kernel {kernel_slug} completed successfully!")
            break
        elif "error" in stdout.lower() or "fail" in stdout.lower():
            print(f"Kernel {kernel_slug} failed with status: {stdout}")
            # Still download output logs for debugging
            run_cmd(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir], check=False)
            return False
            
        time.sleep(poll_interval_sec)
        
    # 3. Download outputs
    print(f"Downloading outputs for {kernel_slug} to {output_dir}...")
    run_cmd(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir])
    print(f"Outputs saved to {output_dir}")
    return True


REGISTERED_KERNELS = [
    "k0-probe",
    "k1-prep",
    "k2-pilot",
    "k3-primary",
    "k4-secondary",
    "k5-controls",
    "k6-mech",
    "k7-latency",
    "k8-m7-v2",
    "k9-freqnoise-v2",
    "k10-resize-ablation",
    "k11-sensor-noise",
    "k12-tome-matched",
    "k13-g0a-and-bn",
]


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "--list"):
        print("Usage: python push_and_wait.py <kernel_name>")
        print("\nRegistered kernels:")
        for k in REGISTERED_KERNELS:
            print(f"  - {k}")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] == "--list" else 1)
        
    kernel_name = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    skip_code_push = "--skip-code-push" in sys.argv
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staging_dir = os.path.join(repo_root, "build", "knobs-code-staging")
    
    kernel_dir = os.path.join(repo_root, "kaggle", "kernels", kernel_name)
    if not os.path.exists(kernel_dir):
        print(f"[ERROR] Unknown kernel '{kernel_name}'. Kernel folder not found at: {kernel_dir}")
        print("Available registered kernels:", ", ".join(REGISTERED_KERNELS))
        sys.exit(1)

    meta_file = os.path.join(kernel_dir, "kernel-metadata.json")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Missing kernel-metadata.json in {kernel_dir}")

    # Push code dataset if not skipped
    if not skip_code_push:
        package_and_push_code_dataset(repo_root, staging_dir)
    else:
        print("Skipping code dataset push as requested (--skip-code-push).")
    
    output_dir = os.path.join(repo_root, "results", "raw", kernel_name)
    
    with open(meta_file, "r") as f:
        meta = json.load(f)
    kernel_slug = meta["id"]
    
    accel = meta.get("accelerator", "NvidiaTeslaT4") if meta.get("enable_gpu", True) else ""
    success = push_and_wait_kernel(kernel_dir, kernel_slug, output_dir, accelerator=accel)
    sys.exit(0 if success else 1)
