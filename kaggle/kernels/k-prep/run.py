"""K-prep: CPU Kaggle kernel for ImageNet validation dataset indexing and split creation.

Costs 0 GPU quota hours.
Creates:
- splits/PILOT.json (5,000 images, 5/class)
- splits/CAL-GATE.json (1,000 images, 1/class)
- splits/CONFIRM_POOL.json (44,000 images, 44/class)
  - splits/PRIMARY_CONFIRM.json (34,000 images, 34/class)
  - splits/SUB10K.json (10,000 images, 10/class)
    - splits/MECH.json (2,000 images, 2/class)
- splits/FULL.json (50,000 images)
- splits/val_metadata.json (metadata mapping for all 50,000 images with bboxes and labels)
"""

import os
import sys
import json
import subprocess

# Dynamically locate knobs package in /kaggle/input
print("Searching for knobs package in /kaggle/input...")
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "knobs" in dirs and os.path.exists(os.path.join(root, "knobs", "__init__.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            print(f"Found knobs package at: {root}, added to sys.path")
            break

import knobs
print(f"Successfully loaded knobs from: {knobs.__file__}")

from knobs.data import create_stratified_splits

# Dynamically locate LOC_val_solution.csv
val_solution_path = None
synset_mapping_path = None

for root, dirs, files in os.walk("/kaggle/input"):
    if "LOC_val_solution.csv" in files:
        val_solution_path = os.path.join(root, "LOC_val_solution.csv")
    if "LOC_synset_mapping.txt" in files:
        synset_mapping_path = os.path.join(root, "LOC_synset_mapping.txt")

if not val_solution_path:
    raise FileNotFoundError("Could not find LOC_val_solution.csv in /kaggle/input")

print(f"Located LOC_val_solution.csv at: {val_solution_path}")
print(f"Located LOC_synset_mapping.txt at: {synset_mapping_path}")

OUTPUT_DIR = "/kaggle/working/splits"
print(f"Writing splits to: {OUTPUT_DIR}")

splits = create_stratified_splits(
    val_solution_path=val_solution_path,
    synset_mapping_path=synset_mapping_path,
    output_dir=OUTPUT_DIR,
    seed=42,
)

summary = {
    name: len(data) for name, data in splits.items()
}
print("Generated split summary:")
print(json.dumps(summary, indent=2))

with open(os.path.join("/kaggle/working", "split_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("K-prep complete!")
