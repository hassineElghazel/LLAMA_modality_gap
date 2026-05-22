#!/usr/bin/env bash
# Download COCO val2017 + COCO train2017 (for LLaVA-Instruct) + Bunny-v1.1
# (Stage 1) + LLaVA-Instruct-150K (Stage 2). Safe to re-run (skips existing).

set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR/coco" "$DATA_DIR/bunny_v1_1" "$DATA_DIR/llava_instruct_150k"

# ---------- COCO val2017 (gap diagnostic + captioning eval) ----------
COCO_DIR="$DATA_DIR/coco"
if [ ! -d "$COCO_DIR/val2017" ]; then
  echo "[data] downloading COCO val2017 (~1 GB)"
  wget -c "http://images.cocodataset.org/zips/val2017.zip" -O "$COCO_DIR/val2017.zip"
  unzip -q "$COCO_DIR/val2017.zip" -d "$COCO_DIR/"
  rm "$COCO_DIR/val2017.zip"
else
  echo "[data] COCO val2017 already present"
fi

# ---------- COCO train2017 (image source for LLaVA-Instruct-150K) ----------
if [ ! -d "$COCO_DIR/train2017" ]; then
  echo "[data] downloading COCO train2017 (~18 GB)"
  wget -c "http://images.cocodataset.org/zips/train2017.zip" -O "$COCO_DIR/train2017.zip"
  unzip -q "$COCO_DIR/train2017.zip" -d "$COCO_DIR/"
  rm "$COCO_DIR/train2017.zip"
else
  echo "[data] COCO train2017 already present"
fi

# ---------- COCO 2017 annotations (val2017 captions) ----------
if [ ! -f "$COCO_DIR/annotations/captions_val2017.json" ]; then
  echo "[data] downloading COCO 2017 annotations"
  wget -c "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
    -O "$COCO_DIR/annotations_trainval2017.zip"
  unzip -q "$COCO_DIR/annotations_trainval2017.zip" -d "$COCO_DIR/"
  rm "$COCO_DIR/annotations_trainval2017.zip"
fi

# ---------- Bunny-v1.1-data (Stage 1 contrastive pretraining) ----------
if [ -z "$(ls -A "$DATA_DIR/bunny_v1_1" 2>/dev/null)" ]; then
  echo "[data] downloading Bunny-v1.1-data from HuggingFace"
  huggingface-cli download BoyaWu10/Bunny-v1_1-data \
    --repo-type dataset \
    --local-dir "$DATA_DIR/bunny_v1_1" \
    --local-dir-use-symlinks False
fi

# ---------- LLaVA-Instruct-150K (Stage 2 autoregressive captioning) ----------
if [ -z "$(ls -A "$DATA_DIR/llava_instruct_150k" 2>/dev/null)" ]; then
  echo "[data] downloading LLaVA-Instruct-150K from HuggingFace"
  huggingface-cli download liuhaotian/LLaVA-Instruct-150K \
    --repo-type dataset \
    --local-dir "$DATA_DIR/llava_instruct_150k" \
    --local-dir-use-symlinks False
fi

echo "[data] done."
