"""End-to-end check of the demo's logic without a browser and without a GPU."""
import sys, time
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import app as A                                              # noqa: E402
from common import CURATED, image_path                       # noqa: E402

fail = 0
def check(name, cond, detail=""):
    global fail
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    fail += (not cond)

be = A.Backend(live=False, timeout=45.0)
print(be.load())

# 1. curated image in cached mode
iid = 85823
img = Image.open(image_path(iid)).convert("RGB")
cap_b, m_b, src_b = be.run(A.BASE, img, iid)
cap_d, m_d, src_d = be.run(A.DRIVE, img, iid)
check("curated: cached source", src_b == src_d == "cached")
check("curated: captions differ", cap_b != cap_d)
check("curated: Cloc_80k scores higher",
      m_d["clipscore"] > m_b["clipscore"], f"{m_b['clipscore']:.3f} -> {m_d['clipscore']:.3f}")
check("curated: caption is the thesis generation",
      cap_b.startswith("The image features a group of horses"), cap_b[:46])
check("curated: metrics show H and recall", "H <b>" in A.metrics_html(m_b, True))

# 2. uploaded image: no ground truth
cap_u, m_u, src_u = be.run(A.DRIVE, img, None)
check("upload: hides H/recall", "no ground-truth annotations" in A.metrics_html(m_u, False))

# 3. red highlighting only marks hallucinated categories
html = A.highlight("a man and a woman on a couch", ["person", "couch"],
                   be.chair.inv, be.chair.singularize)
check("highlight: marks hallucinated (man -> person)",
      '<span class="h">man</span>' in html and '<span class="h">couch</span>' in html)
check("highlight: woman also maps to person", '<span class="h">woman</span>' in html)
html2 = A.highlight("a zebra in a field", ["person"], be.chair.inv, be.chair.singularize)
check("highlight: leaves non-hallucinated alone", '<span class="h">' not in html2)

# 4. live failure falls back to cache for curated images
class Boom(dict):
    def __getitem__(self, k): raise RuntimeError("simulated CUDA OOM")
be_live = A.Backend(live=True, timeout=45.0)
be_live.vlms = Boom({A.DRIVE: None}); be_live.prompt, be_live.gen = "p", {}
cap_f, m_f, src_f = be_live.run(A.DRIVE, img, iid)
check("live failure -> cached fallback", src_f == "cached" and cap_f == cap_d)

# 5. timeout falls back too
be_t = A.Backend(live=True, timeout=0.0)
be_t.prompt, be_t.gen = "p", {}
class Slow:
    def generate(self, *a, **k):
        time.sleep(0.05); return ["x"]
be_t.vlms = {A.DRIVE: Slow()}
cap_to, _, src_to = be_t.run(A.DRIVE, img, iid)
check("timeout -> cached fallback", src_to == "cached")

# 6. every curated image is in the cache for both models
missing = [i for i, _ in CURATED
           if str(i) not in be.cache["entries"]
           or len(be.cache["entries"][str(i)]["models"]) != 2]
check("cache covers all curated images", not missing, str(missing))
print("\nFAILURES:", fail)
sys.exit(1 if fail else 0)
