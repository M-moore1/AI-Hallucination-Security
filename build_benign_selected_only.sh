#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD/benign_selected_only}"
PYTHON="${PYTHON:-python3}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need_cmd curl
need_cmd unzip
need_cmd "$PYTHON"

mkdir -p "$ROOT/meta" "$ROOT/images"/{counting,object_recognition,colors_attributes,spatial_relation,ocr}

fetch() {
  local url="$1"
  local out="$2"
  if [[ ! -f "$out" ]]; then
    echo "Downloading $(basename "$out")"
    curl -L --retry 5 --retry-delay 2 -o "$out" "$url"
  else
    echo "Already have $(basename "$out")"
  fi
}

extract_if_needed() {
  local zipfile="$1"
  local dest="$2"
  local marker="$3"
  mkdir -p "$dest"
  if [[ ! -e "$dest/$marker" ]]; then
    echo "Extracting $(basename "$zipfile")"
    unzip -q -n "$zipfile" -d "$dest"
  else
    echo "Already extracted $(basename "$zipfile")"
  fi
}

# -----------------------------
# Metadata / annotations only
# -----------------------------
fetch "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
      "$ROOT/meta/coco_annotations.zip"

fetch "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip" \
      "$ROOT/meta/vg_image_data.json.zip"
fetch "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip" \
      "$ROOT/meta/vg_attributes.json.zip"
fetch "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships_v1_2.json.zip" \
      "$ROOT/meta/vg_relationships.json.zip"

fetch "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json" \
      "$ROOT/meta/TextOCR_0.1_val.json"

fetch "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py" \
      "$ROOT/meta/openimages_downloader.py"

extract_if_needed "$ROOT/meta/coco_annotations.zip" "$ROOT/meta/coco" "annotations"
extract_if_needed "$ROOT/meta/vg_image_data.json.zip" "$ROOT/meta/vg_image_data" "image_data.json"
extract_if_needed "$ROOT/meta/vg_attributes.json.zip" "$ROOT/meta/vg_attributes" "attributes.json"
extract_if_needed "$ROOT/meta/vg_relationships.json.zip" "$ROOT/meta/vg_relationships" "relationships.json"

# Open Images downloader deps
"$PYTHON" -m pip install --quiet boto3 botocore tqdm

export ROOT
"$PYTHON" - <<'PY'
import csv
import json
import os
import random
import re
from pathlib import Path
from collections import defaultdict

random.seed(42)

ROOT = Path(os.environ["ROOT"])
META = ROOT / "meta"
IMAGES = ROOT / "images"

manifest_rows = []
download_rows = []  # direct URL downloads for COCO + VG
textocr_candidates = []  # OCR candidates for Open Images downloader
used = set()

def add_used(source, image_id):
    key = (source, str(image_id))
    if key in used:
        return False
    used.add(key)
    return True

def add_manifest(category, source, image_id, image_path, prompt, gt):
    manifest_rows.append({
        "category": category,
        "source": source,
        "image_id": str(image_id),
        "image_path": str(image_path),
        "prompt": prompt,
        "ground_truth": gt,
    })

def add_direct_download(url, out_path):
    download_rows.append({"url": url, "out_path": str(out_path)})

# --------------------------------------------------
# COCO: counting + object recognition
# --------------------------------------------------
coco_json = META / "coco" / "annotations" / "instances_val2017.json"
with open(coco_json, "r") as f:
    coco = json.load(f)

images = {img["id"]: img["file_name"] for img in coco["images"]}
cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}

counts_by_image_cat = defaultdict(lambda: defaultdict(int))
total_anns = defaultdict(int)

for ann in coco["annotations"]:
    if ann.get("iscrowd", 0):
        continue
    img_id = ann["image_id"]
    cat_id = ann["category_id"]
    counts_by_image_cat[img_id][cat_id] += 1
    total_anns[img_id] += 1

plural_map = {
    "person": "people",
    "chair": "chairs",
    "car": "cars",
    "bottle": "bottles",
    "banana": "bananas",
    "book": "books",
    "bird": "birds",
    "cup": "cups",
    "bicycle": "bicycles",
    "orange": "oranges",
}

count_targets = [
    "person", "chair", "car", "bottle", "banana",
    "book", "bird", "cup", "bicycle", "orange"
]

for target in count_targets:
    cat_id = cat_name_to_id[target]
    candidates = []
    for img_id, d in counts_by_image_cat.items():
        n = d.get(cat_id, 0)
        if 2 <= n <= 8:
            candidates.append((img_id, n))
    random.shuffle(candidates)
    picked = 0
    for img_id, n in candidates:
        if picked >= 2:
            break
        if not add_used("coco", img_id):
            continue
        file_name = images[img_id]
        url = f"http://images.cocodataset.org/val2017/{file_name}"
        out = IMAGES / "counting" / f"coco_{img_id}.jpg"
        prompt = f"How many {plural_map[target]} are visible in this image? Answer with a number."
        add_direct_download(url, out)
        add_manifest("counting", "coco", img_id, out, prompt, str(n))
        picked += 1

present_targets = [
    "dog", "cat", "bus", "stop sign", "umbrella",
    "laptop", "microwave", "backpack", "horse", "airplane"
]

for target in present_targets:
    cat_id = cat_name_to_id[target]
    candidates = [
        img_id for img_id, d in counts_by_image_cat.items()
        if d.get(cat_id, 0) >= 1
    ]
    random.shuffle(candidates)
    for img_id in candidates:
        if not add_used("coco", img_id):
            continue
        file_name = images[img_id]
        url = f"http://images.cocodataset.org/val2017/{file_name}"
        out = IMAGES / "object_recognition" / f"coco_{img_id}.jpg"
        prompt = f"Is there a {target} in this image? Answer yes or no."
        add_direct_download(url, out)
        add_manifest("object_recognition", "coco", img_id, out, prompt, "yes")
        break

absent_targets = [
    "dog", "cat", "bus", "umbrella", "laptop",
    "backpack", "horse", "airplane", "bicycle", "car"
]

for target in absent_targets:
    cat_id = cat_name_to_id[target]
    candidates = [
        img_id for img_id, d in counts_by_image_cat.items()
        if d.get(cat_id, 0) == 0 and total_anns[img_id] >= 1
    ]
    random.shuffle(candidates)
    for img_id in candidates:
        if not add_used("coco", img_id):
            continue
        file_name = images[img_id]
        url = f"http://images.cocodataset.org/val2017/{file_name}"
        out = IMAGES / "object_recognition" / f"coco_{img_id}.jpg"
        prompt = f"Is there a {target} in this image? Answer yes or no."
        add_direct_download(url, out)
        add_manifest("object_recognition", "coco", img_id, out, prompt, "no")
        break

# --------------------------------------------------
# Visual Genome: colors/attributes + spatial relations
# --------------------------------------------------
with open(META / "vg_image_data" / "image_data.json", "r") as f:
    vg_img_data = json.load(f)
with open(META / "vg_attributes" / "attributes.json", "r") as f:
    vg_attr = json.load(f)
with open(META / "vg_relationships" / "relationships.json", "r") as f:
    vg_rel = json.load(f)

vg_urls = {str(x["image_id"]): x["url"] for x in vg_img_data if "url" in x}

color_words = {
    "red","blue","green","yellow","black","white","brown","orange","pink","purple","gray","grey"
}
allowed_relation_preds = {
    "on","under","behind","in front of","next to","beside","near","above","below","holding","wearing","inside"
}

# 20 color/attribute images (prefer color questions)
attr_candidates = []
for item in vg_attr:
    image_id = str(item.get("image_id"))
    if image_id not in vg_urls:
        continue
    for obj in item.get("attributes", []):
        name = str(obj.get("name", "")).strip().lower()
        attrs = [str(a).strip().lower() for a in obj.get("attributes", [])]
        if not name or not attrs:
            continue
        for a in attrs:
            if a in color_words:
                attr_candidates.append((image_id, name, a, vg_urls[image_id]))
                break

random.shuffle(attr_candidates)
picked_attr = 0
for image_id, obj_name, attr, url in attr_candidates:
    if picked_attr >= 20:
        break
    if not add_used("vg_attr", image_id):
        continue
    out = IMAGES / "colors_attributes" / f"vg_{image_id}.jpg"
    prompt = f"What color is the {obj_name}?"
    add_direct_download(url, out)
    add_manifest("colors_attributes", "visual_genome", image_id, out, prompt, attr)
    picked_attr += 1

# 20 spatial-relation images
rel_candidates = []
for item in vg_rel:
    image_id = str(item.get("image_id"))
    if image_id not in vg_urls:
        continue
    for rel in item.get("relationships", []):
        pred = str(rel.get("predicate", "")).strip().lower()
        if pred not in allowed_relation_preds:
            continue
        subj = str(rel.get("subject", {}).get("name", "")).strip().lower()
        obj = str(rel.get("object", {}).get("name", "")).strip().lower()
        if not subj or not obj:
            continue
        rel_candidates.append((image_id, subj, obj, pred, vg_urls[image_id]))

random.shuffle(rel_candidates)
picked_rel = 0
for image_id, subj, obj, pred, url in rel_candidates:
    if picked_rel >= 20:
        break
    if not add_used("vg_rel", image_id):
        continue
    out = IMAGES / "spatial_relation" / f"vg_{image_id}.jpg"
    prompt = f"Where is the {subj} relative to the {obj}? Answer with a short relation."
    add_direct_download(url, out)
    add_manifest("spatial_relation", "visual_genome", image_id, out, prompt, pred)
    picked_rel += 1

# --------------------------------------------------
# TextOCR: OCR images (selected Open Images IDs only)
# --------------------------------------------------
with open(META / "TextOCR_0.1_val.json", "r") as f:
    tjson = json.load(f)

imgs_raw = tjson.get("imgs", {})
anns_raw = tjson.get("anns", {})

if isinstance(imgs_raw, list):
    img_meta = {str(x["id"]): x for x in imgs_raw}
else:
    img_meta = {str(k): v for k, v in imgs_raw.items()}

if isinstance(anns_raw, list):
    ann_values = anns_raw
else:
    ann_values = anns_raw.values()

clean_word = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]{1,14}$")
texts_by_image = defaultdict(list)

for ann in ann_values:
    txt = ann.get("utf8_string") or ann.get("text") or ann.get("transcription") or ""
    txt = str(txt).strip()
    img_id = ann.get("image_id") or ann.get("img_id") or ann.get("image")
    if img_id is None:
        continue
    img_id = str(img_id)
    if txt and txt not in {".", ",", ":", ";", "-", "_"}:
        texts_by_image[img_id].append(txt)

ocr_candidates = []
for img_id, texts in texts_by_image.items():
    clean = sorted(set(t.strip() for t in texts if clean_word.match(t.strip())))
    if len(clean) != 1:
        continue
    answer = clean[0]
    # TextOCR val images map to Open Images IDs; use train split for train/val images
    ocr_candidates.append((img_id, answer))

random.shuffle(ocr_candidates)
picked_ocr = 0
for img_id, answer in ocr_candidates:
    if picked_ocr >= 20:
        break
    if not add_used("textocr", img_id):
        continue
    out = IMAGES / "ocr" / f"{img_id}.jpg"
    prompt = "What single word is clearly visible in this image? Answer with the exact word."
    textocr_candidates.append((img_id, out, prompt, answer))
    picked_ocr += 1

# Save direct download list
with open(ROOT / "meta" / "direct_downloads.tsv", "w", newline="") as f:
    for row in download_rows:
        f.write(f"{row['url']}\t{row['out_path']}\n")

# Save OCR image list for official Open Images downloader
with open(ROOT / "meta" / "textocr_image_list.txt", "w") as f:
    for img_id, _, _, _ in textocr_candidates:
        f.write(f"train/{img_id}\n")

# Save non-OCR manifest now; OCR rows appended after OCR download
with open(ROOT / "meta" / "manifest_pre_ocr.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["category","source","image_id","image_path","prompt","ground_truth"])
    writer.writeheader()
    writer.writerows(manifest_rows)

# Save OCR planned rows
with open(ROOT / "meta" / "ocr_candidates.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id","image_path","prompt","ground_truth"])
    for img_id, out, prompt, answer in textocr_candidates:
        writer.writerow([img_id, str(out), prompt, answer])

print("Prepared metadata and selection files.")
PY

# -----------------------------
# Download COCO + VG selected images only
# -----------------------------
"$PYTHON" - <<'PY'
import os
import sys
import urllib.request
from pathlib import Path

root = Path(os.environ["ROOT"])
tsv = root / "meta" / "direct_downloads.tsv"

def fetch(url, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    try:
        urllib.request.urlretrieve(url, out)
        print(f"Downloaded {out.name}")
    except Exception as e:
        print(f"FAILED {url} -> {out}: {e}", file=sys.stderr)

with open(tsv, "r") as f:
    for line in f:
        url, out_path = line.rstrip("\n").split("\t")
        fetch(url, out_path)
PY

# -----------------------------
# Download selected OCR images only
# -----------------------------
"$PYTHON" "$ROOT/meta/openimages_downloader.py" \
  "$ROOT/meta/textocr_image_list.txt" \
  --download_folder "$ROOT/images/ocr" \
  --num_processes 8

# -----------------------------
# Finalize manifest with OCR files that actually downloaded
# -----------------------------
"$PYTHON" - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
pre = root / "meta" / "manifest_pre_ocr.csv"
ocr = root / "meta" / "ocr_candidates.csv"
final = root / "manifest.csv"

rows = []
with open(pre, "r", newline="") as f:
    rows.extend(list(csv.DictReader(f)))

with open(ocr, "r", newline="") as f:
    for row in csv.DictReader(f):
        p = Path(row["image_path"])
        if p.exists():
            rows.append({
                "category": "ocr",
                "source": "textocr_openimages",
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "prompt": row["prompt"],
                "ground_truth": row["ground_truth"],
            })

with open(final, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["category","source","image_id","image_path","prompt","ground_truth"])
    writer.writeheader()
    writer.writerows(rows)

# simple summary
from collections import Counter
c = Counter(r["category"] for r in rows)
print("Final counts:")
for k in ["counting","object_recognition","colors_attributes","spatial_relation","ocr"]:
    print(f"  {k}: {c.get(k, 0)}")
print(f"\nManifest: {final}")
PY

echo
echo "Done."
echo "Images are under: $ROOT/images"
echo "Manifest is:      $ROOT/manifest.csv"