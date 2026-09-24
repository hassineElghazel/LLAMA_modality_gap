"""GPU-only: does the loaded model still reproduce the thesis dd256 captions?

Run on the GPU node after the checkpoints are in place. It regenerates five
thesis-figure images with both checkpoints under the dd256 configuration and
diffs against outputs/predictions/*.json. Nothing is tuned to force a match.

    .venv/bin/python demo/regression_live.py
"""
from __future__ import annotations
import difflib, importlib.util, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demo"))
from common import CKPT, MODELS, image_path                        # noqa: E402
from src.utils.io import load_yaml                                 # noqa: E402
from src.data.coco_val2017_loader import load_image                # noqa: E402

IMAGES = [85823, 9914, 113403, 134882, 34760]     # zebras, sandwich, teddy, cat, bathroom


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "demo_caption", ROOT / "scripts" / "30_demo_caption.py")
    dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)

    cfg = load_yaml(ROOT / "configs/description_eval.yaml")
    prompt, gen = cfg["prompt"]["user"].strip(), dict(cfg["generation"])
    enc = load_yaml(ROOT / "configs/encoders.yaml")
    proj = load_yaml(ROOT / "configs/projector.yaml")
    llm = load_yaml(ROOT / "configs/llm.yaml")
    lora = load_yaml(ROOT / "configs/training_stage2.yaml").get("lora")
    print(f"prompt   : {prompt!r}\ngeneration: {gen}\n")

    bad = 0
    for label, ck in CKPT.items():
        ref = {d["image_id"]: d["caption"] for d in
               json.loads((ROOT / "outputs/predictions" / MODELS[label]).read_text())}
        t0 = time.time()
        vlm = dc.build(str(ROOT / ck), enc, proj, llm, lora)
        print(f"== {label}   load {time.time()-t0:.0f}s")
        for iid in IMAGES:
            src = image_path(iid)
            img = load_image(str(src))
            t1 = time.time()
            got = vlm.generate([img], [prompt], **gen)[0].strip()
            dt = time.time() - t1
            want = ref[iid].strip()
            same = got == want
            bad += (not same)
            ratio = difflib.SequenceMatcher(None, got, want).ratio()
            print(f"   {iid:7d}  {'MATCH' if same else 'DIFF '}  "
                  f"similarity {ratio:.3f}  {dt:5.1f}s  {len(got.split()):3d} words"
                  f"  [{'coco' if 'val2017' in str(src) else 'bundled'}]")
            if not same:
                print(f"      thesis: {want[:150]}")
                print(f"      demo  : {got[:150]}")
        del vlm
        import torch; torch.cuda.empty_cache()
    print(f"\n{len(IMAGES)*2 - bad}/{len(IMAGES)*2} exact matches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
