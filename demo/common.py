"""Shared pieces for the defence demo: curated set, CHAIR fields, CLIPScore.

Everything here reuses the thesis evaluation code rather than reimplementing it:
CHAIR comes from scripts/19_chair.py (same synonym list, same double-word rules,
same GT = segmentation categories UNION caption-derived objects), and CLIPScore
uses the ViT-B/32 backbone with w = 2.5 that scripts/15_clipscore.py uses.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"

MODELS = {                                   # label -> thesis prediction file
    "all-pinned (baseline)": "captions_C3pinr_dd256.json",
    "Cloc_80k (location drive)": "captions_Cloc_80k_dd256.json",
}
CKPT = {                                     # label -> checkpoint, for --live
    "all-pinned (baseline)": "outputs/checkpoints/stage2_vlm_C3pinr.pt",
    "Cloc_80k (location drive)": "outputs/checkpoints/stage2_vlm_Cloc_80k.pt",
}

# curated COCO val2017 images; the thesis figure examples first
CURATED = [
    (85823,  "Zebras  (thesis Fig. 4.8)"),
    (113403, "Teddy bears  (Fig. 4.19)"),
    (9914,   "Sandwich and fries  (Fig. 4.7)"),
    (134882, "Cat on a bed  (Fig. 4.20)"),
    (34760,  "Bathroom  (Fig. 4.21)"),
    (92839,  "Polar bear  (Fig. 4.11)"),
    (135890, "Clock  (Fig. 4.9)"),
    (99182,  "Desk  (Fig. 4.10)"),
    (3156,   "COCO val2017"),
    (19432,  "COCO val2017"),
    (42070,  "COCO val2017"),
    (109118, "COCO val2017"),
]

COCO_VAL = ROOT / "data" / "coco" / "val2017"


def image_path(image_id: int) -> Path:
    """Prefer the true COCO val2017 file; fall back to the bundled copy.

    Two of the bundled thesis-figure JPEGs (85823, 99182) are re-encoded and
    score ~0.02 CLIPScore below the thesis value, so the original always wins
    where it is available (i.e. on the cluster).
    """
    orig = COCO_VAL / f"{image_id:012d}.jpg"
    return orig if orig.exists() else DEMO / "images" / f"{image_id:012d}.jpg"


def _chair_module():
    spec = importlib.util.spec_from_file_location(
        "chair19", ROOT / "scripts" / "19_chair.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Chair:
    """Hallucination fields for one caption, identical to the thesis pipeline."""

    def __init__(self):
        ch = _chair_module()
        self._c2n = ch.caption_to_node_words
        self.singularize = ch.singularize
        self.objects, self.inv = ch.load_synonyms(ROOT / "scripts" / "chair_synonyms.txt")
        self.dwd = ch._build_double_word_dict()
        refs: dict[int, list[str]] = {}
        caps = json.loads((ROOT / "data/coco/annotations/captions_val2017.json").read_text())
        for a in caps["annotations"]:
            refs.setdefault(a["image_id"], []).append(a["caption"])
        ids = [i for i, _ in CURATED]
        self.gt, self.gt_source = ch.build_gt(
            ids, refs, self.objects, self.inv, self.dwd,
            ROOT / "data/coco/annotations/instances_val2017.json")

    def score(self, image_id: int, caption: str) -> dict:
        nodes, n_tokens = self._c2n(caption, self.objects, self.inv, self.dwd)
        gt = self.gt.get(image_id, set())
        named = sorted(set(nodes))
        hallucinated = sorted({w for w in nodes if w not in gt})
        # H is n_hall_words from 19_chair.py::score_condition: hallucinated
        # MENTIONS, counted with multiplicity, not distinct categories.
        return {
            "words": n_tokens,
            "named": named,
            "hallucinated": hallucinated,
            "H": sum(1 for w in nodes if w not in gt),
            "recall": (len(set(nodes) & gt) / len(gt)) if gt else None,
            "n_gt": len(gt),
            "gt": sorted(gt),
        }


class ClipScorer:
    """CLIPScore = 2.5 * max(cos(img, txt), 0), ViT-B/32, as in 15_clipscore.py."""

    MODEL = "openai/clip-vit-base-patch32"
    W = 2.5

    def __init__(self, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model = CLIPModel.from_pretrained(self.MODEL).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(self.MODEL)

    def score(self, image, caption: str) -> float:
        t = self.torch
        with t.no_grad():
            ii = self.proc(images=image, return_tensors="pt").to(self.device)
            ie = self.model.get_image_features(**ii)
            ti = self.proc(text=[caption], return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(self.device)
            te = self.model.get_text_features(**ti)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            cos = float((ie * te).sum(-1).item())
        return self.W * max(cos, 0.0)
