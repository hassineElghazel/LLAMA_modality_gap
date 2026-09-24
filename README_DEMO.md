# Defence demo — all-pinned vs Cloc_80k

One page, two caption panels, same image. Left is the pinned baseline, right is
the location-driven model at 2,500 steps. Everything shown traces to the thesis.

## Two modes

| mode | needs | what it shows |
|---|---|---|
| `--cached` | nothing (laptop is fine) | replays `demo/cache.json` with a simulated token stream. Captions are the **real thesis dd256 generations**, never hand-written. A small `cached` tag is shown on every panel. |
| `--live` | CUDA, both checkpoints | loads both models and generates for real. Falls back to the cache on any error or after `--timeout` seconds (default 45). |

## Launch

```bash
# laptop, no GPU
./demo/launch_cached.sh                  # http://127.0.0.1:7860

# GPU node, inside an allocation
srun --partition=RTX --gpus=2080ti:1 --pty bash
./demo/launch_gpu.sh                     # binds 127.0.0.1:7860 on the node
```

From your laptop, tunnel to it and open `http://localhost:7860`:

```bash
ssh -N -L 7860:localhost:7860 <user>@<gpu-node>
```

If SSH forwarding is awkward on the day, `--share` gives a public Gradio URL
instead (`.venv/bin/python demo/app.py --live --share`). It routes through
Gradio's relay, so prefer the tunnel.

## Requirements

The repo `.venv` plus `gradio>=5,<6`. **`huggingface-hub` must stay `<1.0`** —
`transformers==4.44.2` refuses to import otherwise, which breaks the whole
evaluation stack. Install with the pin:

```bash
.venv/bin/pip install "gradio>=5,<6" "huggingface-hub>=0.23.2,<1.0"
```

## Rebuilding the cache

```bash
# on the cluster, against the true COCO originals (preferred)
.venv/bin/python demo/build_cache.py --image-root data/coco/val2017
```

Captions come from `outputs/predictions/captions_{C3pinr,Cloc_80k}_dd256.json`.
CLIPScore and CHAIR are recomputed by the demo's own scorers, so cached and live
panels agree by construction. Provenance is recorded inside `cache.json`.

## Tests

```bash
.venv/bin/python demo/selftest.py            # 12 checks, no GPU
.venv/bin/python demo/regression_scoring.py  # CLIPScore + H vs every thesis figure
.venv/bin/python demo/regression_live.py     # GPU: captions vs thesis dd256
```

## Configuration (fixed, not exposed in the UI)

Loaded from `configs/description_eval.yaml` — the dd256 setting that produced the
thesis captions: prompt `"Describe the following image in detail."`,
`do_sample: false`, `num_beams: 1`, `max_new_tokens: 256`. CLIP ViT-L/14 @224,
257 visual tokens, 2-layer MLP connector, LLaMA-2-7B 4-bit NF4, LoRA r=8.
CLIPScore uses ViT-B/32 with w = 2.5, the thesis's non-circular setting.

## Known limits

- Both models are loaded and run **sequentially**; the status line says which is
  generating. They share nothing at present — see "Untested" below.
- Uploaded images show CLIPScore and word count only. H and recall need COCO
  annotations, so those fields are replaced by "no ground-truth annotations".
- `demo/images/` holds the thesis figure copies. Two of them
  (`85823`, `99182`) are **re-encoded JPEGs**, not COCO originals, and score
  ~0.02 CLIPScore below the thesis value. Rebuild the cache with
  `--image-root data/coco/val2017` on the cluster to fix this.

## Screen recording script (60–90 s backup)

1. Start cached mode, wait for the status line to read **ready** (0:00–0:05).
2. Click **Zebras** — let both panels stream. Point at `horses` → `zebras`
   and CLIPScore 0.538 → 0.755 (0:05–0:30).
3. Click **Teddy bears** — baseline invents a living room, H 13 → 0,
   CLIP 0.429 → 0.868 (0:30–0:55).
4. Click **Bathroom** — CLIP 0.385 → 0.809, H 10 → 0 (0:55–1:15).
5. Drag any photo onto the upload box to show it also works on unseen images,
   with the ground-truth fields correctly hidden (1:15–1:30).
