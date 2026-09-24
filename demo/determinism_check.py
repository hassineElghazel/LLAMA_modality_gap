"""Is the demo reproducible? Separates two possible causes of drift.

  A. within one load  : same model object, same image, 3 times.
     Differences here would mean sampling is on (it is not) or a kernel is
     genuinely non-deterministic.
  B. across two loads : rebuild the model, caption again.
     Differences here but not in A mean load-time randomness -- almost certainly
     resize_token_embeddings(mean_resizing=True), which draws the new <image>
     row from a multivariate normal on every load.

Run on the GPU node:  python demo/determinism_check.py
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demo"))
from common import CKPT, image_path                          # noqa: E402
from src.utils.io import load_yaml                           # noqa: E402
from src.data.coco_val2017_loader import load_image          # noqa: E402

IID = 113403
LABEL = "Cloc_80k (location drive)"


def build_once(dc, cfgs, seed=None):
    if seed is not None:
        import torch, random, numpy as np
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
    return dc.build(str(ROOT / CKPT[LABEL]), *cfgs)


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "demo_caption", ROOT / "scripts" / "30_demo_caption.py")
    dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)
    cfg = load_yaml(ROOT / "configs/description_eval.yaml")
    prompt, gen = cfg["prompt"]["user"].strip(), dict(cfg["generation"])
    cfgs = (load_yaml(ROOT / "configs/encoders.yaml"),
            load_yaml(ROOT / "configs/projector.yaml"),
            load_yaml(ROOT / "configs/llm.yaml"),
            load_yaml(ROOT / "configs/training_stage2.yaml").get("lora"))
    img = load_image(str(image_path(IID)))

    print("A. three generations from ONE load")
    vlm = build_once(dc, cfgs, seed=42)
    a = []
    for i in range(3):
        t = time.time(); a.append(vlm.generate([img], [prompt], **gen)[0].strip())
        print(f"   run {i+1}: {len(a[-1].split()):3d} words  {time.time()-t:4.1f}s")
    print("   all identical:", len(set(a)) == 1)
    del vlm
    import torch; torch.cuda.empty_cache()

    print("\nB. one generation from a SECOND load (same seed)")
    vlm = build_once(dc, cfgs, seed=42)
    b = vlm.generate([img], [prompt], **gen)[0].strip()
    print("   matches A:", b == a[0])
    del vlm; torch.cuda.empty_cache()

    print("\nC. one generation from a THIRD load (different seed)")
    vlm = build_once(dc, cfgs, seed=1234)
    c = vlm.generate([img], [prompt], **gen)[0].strip()
    print("   matches A:", c == a[0])

    print("\nverdict:")
    if len(set(a)) != 1:
        print("  within-load non-determinism -> kernel level, seeding will not fix it")
    elif b != a[0]:
        print("  differs across loads even at the same seed -> not the RNG;")
        print("  4-bit kernel / library version drift")
    elif c != a[0]:
        print("  same seed reproduces, different seed does not ->")
        print("  load-time RNG (resize_token_embeddings). Seed the demo and it is stable.")
    else:
        print("  fully reproducible; the thesis mismatch is library-version drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
