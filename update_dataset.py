import os
import shutil
import time
import subprocess

repo_root = os.path.dirname(os.path.abspath(__file__))
staging = os.path.join(repo_root, "build", "knobs-code-staging")
src_dest = os.path.join(staging, "src")
if os.path.exists(src_dest):
    shutil.rmtree(src_dest)
shutil.copytree(os.path.join(repo_root, "src"), src_dest)
print("Updated src in staging.")

msg = f"Update code {time.strftime('%Y-%m-%d %H:%M:%S')}"
cmd = ["kaggle", "datasets", "version", "-p", staging, "-m", msg, "--dir-mode", "zip"]
print("Running:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)
if res.returncode != 0:
    raise RuntimeError(f"Dataset version failed with code {res.returncode}")
print("Dataset version successfully pushed.")
