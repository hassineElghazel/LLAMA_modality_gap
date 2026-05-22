# Measuring the Modality Gap in MLLMs — Connector Ablation Study

> Master thesis project · Data Science & Engineering, a.y. 2025–2026

---

## Objective

Measure how the connector's training regime affects the modality gap in a Multimodal Large Language Model.
Through four experimental conditions, we isolate the contributions of contrastive pre-training (Stage 1)
and autoregressive refinement (Stage 2) to the evolution of the modality gap. All measurements are taken
in the LLM's 4096-dim input space.

---

## Architecture

The pipeline is fixed across all conditions. A single connector maps CLIP visual features into LLaMA's input space.

```
 ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
 │    Image     │────▶│  CLIP ViT-L/14   │────▶│   Connector      │────▶│  LLaMA-2-7B      │
 │   224×224    │     │    [FROZEN]       │     │      MLP         │     │   + LoRA         │
 └──────────────┘     │     → 1024       │     │  1024 → 4096     │     │    4096          │
                      └──────────────────┘     └────────┬─────────┘     └──────────────────┘
                                                         │
                                                ╔════════╧════════╗
                                                ║  Measurement Pt ║
                                                ║   4096-dim      ║
                                                ╚═════════════════╝
```

**Figure 1:** Full pipeline. The connector is the only component that varies across conditions.
The modality gap is measured at the connector output.

| Component | Architecture | Dimension | Role |
|:---|:---|:---|:---|
| CLIP ViT-L/14 | Vision transformer | → 1024 | Frozen vision encoder |
| Connector | 2-layer MLP + GELU | 1024 → 4096 | Maps visual features to LLM space |
| LLaMA-2-7B | Causal LM + LoRA | 4096 | Text generation |

---

## Stage 1: Contrastive Connector Pre-training

Stage 1 is a standalone preparation step. It trains the connector with contrastive loss so that it does
not start with random weights in Stage 2. The alignment happens in LLaMA's 4096-dim space.

```
 ┌────────┐   ┌──────────────────┐   ┌──────────────────────┐
 │ Image  │──▶│ CLIP ViT-L/14   │──▶│  Connector           │──▶  z_img
 └────────┘   │   [FROZEN]      │   │  1024 → 4096         │
              └──────────────────┘   │  [TRAINED]           │
                                     └──────────┬───────────┘
                                                │
                                       ╔════════╧═════════╗          z_img ──┐
                                       ║  Measure gap     ║                  ├──▶  InfoNCE Loss
                                       ║  after training  ║          z_txt ──┘
                                       ╚══════════════════╝
 ┌──────────┐   ┌─────────────────┐   ┌──────────────────────┐
 │ Caption  │──▶│ LLaMA Tokenizer │──▶│  LLaMA Embed         │──▶  z_txt
 └──────────┘   └─────────────────┘   │  [FROZEN] → 4096     │
                                      └──────────────────────┘
```

**Figure 2:** Stage 1 — contrastive alignment in LLaMA's 4096-dim space.

### Training Setup

| Component | Status |
|:---|:---|
| CLIP ViT-L/14 | Frozen |
| Connector (1024 → 4096) | **Trained** |
| LLaMA Embed Layer | Frozen (text-side target only) |
| LLaMA (rest of model) | Not involved |

### Forward Pass

- **Image side:** image → Frozen ViT → CLS token (1024) → Connector → **z_img** (4096)
- **Text side:** caption → LLaMA tokenizer → LLaMA embed layer (frozen) → mean pool → **z_txt** (4096)

### Loss Function

Symmetric InfoNCE loss over a batch of $N$ image-caption pairs:

$$\mathcal{L} = \frac{1}{2}\left[\mathrm{CE}\!\left(\frac{\mathbf{z}_{\mathrm{img}} \cdot \mathbf{z}_{\mathrm{txt}}^\top}{\tau},\,\mathbf{y}\right) + \mathrm{CE}\!\left(\frac{\mathbf{z}_{\mathrm{txt}} \cdot \mathbf{z}_{\mathrm{img}}^\top}{\tau},\,\mathbf{y}\right)\right]$$

where $\tau$ is a learnable temperature (initialised at $0.07$) and $\mathbf{y} = [0, 1, \ldots, N{-}1]$.

### Data

`BoyaWu10/Bunny-v1.1-data`

### Output

Saved connector checkpoint.

---

## Stage 2: Autoregressive Image Captioning

Stage 2 refines the connector and trains the LLM to generate text conditioned on visual tokens.

```
 ┌────────┐   ┌────────────────┐   ┌────────────────────────┐
 │ Image  │──▶│  ViT [FROZEN]  │──▶│  Connector [REFINED]   │──▶ 257 × 4096 ──┐
 └────────┘   └────────────────┘   │  1024 → 4096           │                  │
                                   └──────────┬─────────────┘                  ├──▶ [LLaMA-2-7B + LoRA] ──▶ Caption
                                              │                                  │        [TRAINED]
                                     ╔════════╧═════════╗    Caption tokens ───┘
                                     ║  Measure gap     ║    (tokenize + embed)
                                     ║  after training  ║
                                     ╚══════════════════╝
```

**Figure 3:** Stage 2 — autoregressive captioning. The connector is refined, not frozen.

### Training Setup

| Component | Status |
|:---|:---|
| CLIP ViT-L/14 | Frozen |
| Connector (1024 → 4096) | **Refined** (loaded from Stage 1 or random) |
| LLaMA-2-7B + LoRA | **Trained** |

### Forward Pass

**Image side:**

$$\text{image} \;\to\; \text{Frozen ViT} \;\to\; 257 \times 1024 \;\to\; \text{Connector (refined)} \;\to\; 257 \times 4096 \quad \text{(visual tokens)}$$

**Text side:**

$$\text{caption} \;\to\; \text{LLaMA tokenizer} \;\to\; \text{LLaMA embed layer} \;\to\; \mathrm{seq\_len} \times 4096 \quad \text{(text tokens)}$$

**Combined:** Visual tokens and text tokens are concatenated. LLaMA (+ LoRA) predicts the next token
autoregressively. Cross-entropy loss on text positions only.

### Data

`liuhaotian/LLaVA-Instruct-150K`

---

## Experimental Conditions

Four conditions isolate the contributions of Stage 1 (contrastive) and Stage 2 (autoregressive)
to the modality gap.

### C0: No Training (Baseline)

- Connector: random initialisation
- LLaMA: pre-trained weights, no LoRA
- **Nothing is trained. Stage 1 is not run. Stage 2 is not run.**
- Measure the modality gap

### C1: Stage 2 Only (No Contrastive Preparation)

- Stage 1 is not run
- Connector: starts random, **trained** directly in Stage 2
- LLaMA + LoRA: **trained**
- Run Stage 2 → measure the modality gap after Stage 2

### C2: Stage 1 Only (No Autoregressive Refinement)

- Run Stage 1: train connector with contrastive loss → save checkpoint
- Measure the modality gap after Stage 1
- Plug connector into LLM. **Stage 2 is not run.**
- Evaluate zero-shot captioning performance

### C3: Full Pipeline (Stage 1 + Stage 2)

- Run Stage 1: train connector with contrastive loss → save checkpoint
- Measure the modality gap after Stage 1
- Load trained connector into Stage 2; connector **refined** during Stage 2; LLaMA + LoRA **trained**
- Run Stage 2 → measure the modality gap after Stage 2

### Conditions Summary

| | Connector init | Stage 1 (contrastive) | Stage 2 (autoregressive) |
|:---|:---:|:---:|:---:|
| **C0** | Random | ✗ | ✗ |
| **C1** | Random | ✗ | ✓ |
| **C2** | Random | ✓ | ✗ |
| **C3** | Random | ✓ | ✓ |

---

## Modality Gap Measurement

All measurements are taken at the **connector output** (4096-dim), comparing image embeddings from the
connector against text embeddings from LLaMA's embedding layer. The same evaluation set is used for
every measurement.

### Embeddings Compared

$$\mathbf{z}_{\mathrm{img}} = \mathrm{mean\text{-}pool}\!\bigl(\mathrm{Connector}(\mathrm{ViT}(\mathrm{image}))\bigr) \in \mathbb{R}^{4096}$$

$$\mathbf{z}_{\mathrm{txt}} = \mathrm{mean\text{-}pool}\!\bigl(\mathrm{LLaMA\_embed}(\mathrm{tokenize}(\mathrm{caption}))\bigr) \in \mathbb{R}^{4096}$$

### Measurement Schedule

| When | C0 | C1 | C2 | C3 |
|:---|:---:|:---:|:---:|:---:|
| At initialisation (random connector) | ✓ | — | — | — |
| After Stage 1 (contrastive) | — | — | ✓ | ✓ |
| After Stage 2 (autoregressive) | — | ✓ | — | ✓ |

This produces **5 gap measurements**: 1 from C0, 1 from C1, 1 from C2, and 2 from C3.

### Key Comparisons

| Comparison | Variable isolated | Question answered |
|:---|:---|:---|
| C0 vs C2 | Stage 1 alone | Does contrastive training reduce the gap? |
| C0 vs C1 | Stage 2 alone | Does autoregressive training reduce the gap? |
| C1 vs C3 | Effect of Stage 1 preparation | Does contrastive pre-training help Stage 2? |
| C2 vs C3 | Effect of Stage 2 refinement | Does autoregressive refinement further close the gap? |

The **core comparison is C1 vs C3**. If C3's gap is significantly smaller than C1's, then contrastive
pre-training (Stage 1) provides a measurable benefit that autoregressive training alone cannot achieve.
If they are similar, Stage 1 was unnecessary.

---

## Metrics

All metrics are computed at the connector output (4096-dim) at each measurement point in the schedule above.

### Geometric Metrics

| Metric | What it measures |
|:---|:---|
| Centroid distance | First-order gap magnitude |
| Power-law exponent α | Semantic hierarchy preservation |
| JS divergence | Angular topology mismatch |
| k-NN mixing rate | Manifold penetration |
| ‖β‖, ‖γ‖ | Bias decomposition (PMB, COB) |
| κ(Σ_U), κ(Σ_V) | Residual anisotropy |
| Effective rank | Representation compactness |
| Trace | Global variance scale |

### Downstream Performance

Evaluated for conditions C1, C2, and C3 (C0 has no trained LLM).

| Category | Benchmarks |
|:---|:---|
| General Perception | MME, MMStar, ScienceQA-image, RealWorldQA |
| Complex Reasoning | MMMU, MMMU-Pro, VisuLogic, LogicVista |
| Hallucination | CRPE, POPE, HallusionBench |
| Captioning Quality | CIDEr, BLEU-4, METEOR (COCO val2017) |

### Visualisation

PCA visualisations of the combined image and text embedding distributions (4096-dim) for all conditions,
projecting onto the first two principal components. Additionally, gap decomposition plots showing how the
bias (β, γ) and residual anisotropy (κ) evolve from C0 through C2 to C3.

### Evaluation Data

COCO val2017: 5,000 images with 5 captions each. This is the evaluation set used for every measurement
point across all conditions. No overlap with Stage 1 or Stage 2 training data.

---

## Setup

### Prerequisites

- Python 3.10
- CUDA 12.1+ recommended; bf16-capable GPU
- HuggingFace account with access to `meta-llama/Llama-2-7b-hf` (gated)
- ~150 GB free disk for datasets and checkpoints

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"
bash scripts/01_download_data.sh
```

---

## Running the Ablation

Each condition is fully orchestrated by `scripts/run_condition.py`:

```bash
python scripts/run_condition.py --condition C0   # random connector — gap only
python scripts/run_condition.py --condition C1   # Stage 2 from random connector
python scripts/run_condition.py --condition C2   # Stage 1 only, zero-shot LLM
python scripts/run_condition.py --condition C3   # Stage 1 → Stage 2
```

Per-step invocation:

```bash
# Stage 1 InfoNCE pretraining
python scripts/05_train_stage1.py --config configs/training_stage1.yaml

# Stage 2 LoRA SFT (init from Stage 1 connector for C3, from random for C1)
python scripts/06_train_stage2.py --config configs/training_stage2.yaml \
       --stage1-checkpoint outputs/checkpoints/stage1_connector.pt

# Gap measurement per condition
python scripts/07_extract_projected.py --condition C3_stage1
python scripts/03_compute_gap.py --condition C3_stage1
python scripts/04_make_plots.py --condition C3_stage1

# Downstream
python scripts/08_run_captioning.py --condition C3_stage2
python scripts/09_score_captions.py --condition C3_stage2
python scripts/10_run_vlmevalkit.py --condition C3_stage2
```

### Tests

```bash
pytest tests/ -v                            # full suite (95 tests)
pytest tests/ -m "not slow and not gpu"     # fast CI subset
```

---

## Repository Layout

```
configs/
  encoders.yaml             CLIP ViT-L/14 @ 224 config
  projector.yaml            MLP connector: 1024 → 4096 → 4096
  llm.yaml                  LLaMA-2-7B config
  data.yaml                 Bunny-v1.1 · LLaVA-Instruct-150K · COCO val2017
  captioning.yaml           Generation + COCO val2017 eval config
  training_stage1.yaml      InfoNCE schedule, learnable τ
  training_stage2.yaml      LoRA targets, hyperparameters

src/
  encoders/clip_encoder.py  CLIP ViT-L/14 → (B, 257, 1024)
  models/projector.py       MLP2xGELU (1024 → 4096 → 4096)
  models/vlm.py             Encoder + connector + LLaMA-2 splice
  data/coco_val2017_loader.py
  data/bunny_v1_1_loader.py
  data/llava_instruct_loader.py
  training/contrastive_loss.py   Symmetric InfoNCE with learnable τ
  training/stage1_pretrain.py
  training/stage2_sft.py
  diagnostics/metrics.py    Spec metrics + Float64 discipline
  diagnostics/plots.py      Per-condition figures + trajectory
  evaluation/vlmevalkit_adapter.py

scripts/
  01_download_data.sh
  05_train_stage1.py  ·  06_train_stage2.py
  07_extract_projected.py   --condition {C0,C1,C2,C3-stage1,C3-stage2}
  08_run_captioning.py  ·  09_score_captions.py  ·  10_run_vlmevalkit.py
  run_condition.py          End-to-end orchestrator
```

---

## Reproducibility

Every script writes alongside its outputs a config snapshot, git commit hash, `pip freeze` dump, random
seeds, hardware info, and walltime. `torch.backends.cudnn.deterministic = True` and `benchmark = False`
are set globally.

---

## References

```bibtex
@article{yu2026realign,
  title   = {ReAlign: Addressing the Modality Gap in Multimodal LLMs},
  author  = {Yu, Xiao-Ming and others},
  journal = {arXiv preprint arXiv:2602.07026},
  year    = {2026}
}

@article{yu2026aniso,
  title   = {AnisoAlign: Anisotropic Residual Structure in Multimodal Alignment},
  author  = {Yu, Xiao-Ming and others},
  journal = {arXiv preprint arXiv:2605.07825},
  year    = {2026}
}

@article{liang2022modality,
  title   = {Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive
             Representation Learning},
  author  = {Liang, Weixin and others},
  journal = {NeurIPS},
  year    = {2022}
}

@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and others},
  booktitle = {ICML},
  year      = {2021}
}

@article{touvron2023llama2,
  title   = {Llama 2: Open Foundation and Fine-Tuned Chat Models},
  author  = {Touvron, Hugo and others},
  journal = {arXiv preprint arXiv:2307.09288},
  year    = {2023}
}

@inproceedings{liu2023llava,
  title     = {Visual Instruction Tuning},
  author    = {Liu, Haotian and others},
  booktitle = {NeurIPS},
  year      = {2023}
}

@misc{vlmevalkit,
  title  = {VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models},
  author = {OpenCompass Contributors},
  year   = {2024},
  url    = {https://github.com/open-compass/VLMEvalKit}
}
```

---

## License

MIT (project code). Vendored third-party code retains its original license — see `THIRD_PARTY_LICENSES.md`.
