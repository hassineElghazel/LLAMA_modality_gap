#!/usr/bin/env bash
# Download COCO 2014 + Karpathy split + Bunny-pretrain + InternVL-Chat-V1.2-SFT.
# Per plan §6. Total disk: ~200 GB. Network-heavy; safe to re-run (skips
# existing files).

set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR/coco" "$DATA_DIR/karpathy" "$DATA_DIR/bunny_pretrain" "$DATA_DIR/internvl_sft"

# ---------- COCO 2014 ----------
COCO_DIR="$DATA_DIR/coco"
for split in train2014 val2014; do
  if [ ! -d "$COCO_DIR/$split" ]; then
    echo "[data] downloading COCO $split (~13 GB)"
    wget -c "http://images.cocodataset.org/zips/${split}.zip" -O "$COCO_DIR/${split}.zip"
    unzip -q "$COCO_DIR/${split}.zip" -d "$COCO_DIR/"
    rm "$COCO_DIR/${split}.zip"
  else
    echo "[data] COCO $split already present"
  fi
done

# ---------- Karpathy split ----------
KARP_JSON="$DATA_DIR/karpathy/dataset_coco.json"
if [ ! -f "$KARP_JSON" ]; then
  echo "[data] downloading Karpathy split"
  # Common mirror — confirm checksum at first run.
  wget -c "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip" \
    -O "$DATA_DIR/karpathy/caption_datasets.zip"
  unzip -q "$DATA_DIR/karpathy/caption_datasets.zip" -d "$DATA_DIR/karpathy/"
  rm "$DATA_DIR/karpathy/caption_datasets.zip"
fi

# ---------- Bunny-pretrain 1M ----------
if [ -z "$(ls -A "$DATA_DIR/bunny_pretrain" 2>/dev/null)" ]; then
  echo "[data] downloading Bunny-pretrain 1M from HuggingFace"
  huggingface-cli download BoyaWu10/Bunny-v1_0-data \
    --repo-type dataset \
    --local-dir "$DATA_DIR/bunny_pretrain" \
    --local-dir-use-symlinks False
fi

# ---------- InternVL-Chat-V1.2-SFT ----------
if [ -z "$(ls -A "$DATA_DIR/internvl_sft" 2>/dev/null)" ]; then
  echo "[data] InternVL-Chat-V1.2-SFT must be downloaded manually from the InternVL release."
  echo "[data] Place the data at $DATA_DIR/internvl_sft/ and rerun."
fi

echo "[data] done."
