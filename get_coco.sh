#!/usr/bin/env bash
set -euo pipefail

COUNT="${1:-40}"

python3 -m pip install -q datasets pillow

python3 - "$COUNT" <<'PY'
import os
import sys
from itertools import islice
from datasets import load_dataset

count = int(sys.argv[1])
outdir = os.path.abspath("./coco")
os.makedirs(outdir, exist_ok=True)

# Stream COCO val split without downloading the full dataset
ds = load_dataset("detection-datasets/coco", split="val", streaming=True)

saved = 0
for ex in islice(ds, count):
    img = ex["image"]
    image_id = ex.get("image_id", saved)
    img.save(os.path.join(outdir, f"{image_id}.jpg"))
    saved += 1

print(f"Saved {saved} images to {outdir}")
PY