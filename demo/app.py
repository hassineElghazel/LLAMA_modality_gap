"""Defence demo: all-pinned vs Cloc_80k, side by side, on one page.

Two modes, chosen at launch:

  --cached   replay demo/cache.json with a simulated token stream. No GPU.
             Every caption in the cache is a real thesis generation.
  --live     load both checkpoints on a CUDA box and generate for real.
             Falls back to the cache on error or after --timeout seconds.

    python demo/app.py --cached                      # laptop
    python demo/app.py --live                         # GPU node
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from common import (CURATED, MODELS, CKPT, Chair, ClipScorer, image_path,
                    build_vlm)                                    # noqa: E402

BASE, DRIVE = list(MODELS)                      # panel order, left then right
ACCENT, INK, MUTE, LINE = "#c06a3e", "#262626", "#707070", "#dcdcd8"
FOOTER = "Cloc_80k: 2,500 steps · 80,000 items · single RTX 2080 Ti"

CSS = f"""
.gradio-container {{ background:#f5f5f3 !important; font-family:'Quicksand',
  'Helvetica Neue',sans-serif; max-width:1400px !important; }}
#hdr h1 {{ color:{INK}; font-weight:600; letter-spacing:.06em; margin:0 0 2px; }}
#hdr p {{ color:{MUTE}; margin:0; font-size:.86rem; }}
.panel {{ background:#fff; border:1px solid {LINE}; border-radius:10px;
  padding:14px 16px; }}
.panel h3 {{ margin:0 0 8px; font-size:.95rem; color:{INK}; font-weight:700; }}
.cap {{ font-size:1.0rem; line-height:1.5; color:{INK}; min-height:118px; }}
.cap .h {{ color:#aa1423; }}
.metrics {{ margin-top:10px; padding-top:9px; border-top:1px solid {LINE};
  font-size:.84rem; color:{MUTE}; }}
.metrics b {{ color:{ACCENT}; font-size:.96rem; }}
#status {{ font-size:.8rem; color:{MUTE}; }}
.hint {{ font-size:.76rem; color:#8a8a86; margin:8px 0 2px; }}
#foot {{ font-size:.76rem; color:#a1a1a1; text-align:center; margin-top:10px; }}
.tag {{ font-size:.68rem; color:#fff; background:{MUTE}; border-radius:3px;
  padding:1px 6px; margin-left:6px; vertical-align:2px; }}
"""


def highlight(caption: str, words: list[str], inv, singularize=None) -> str:
    """Red-tint surface tokens whose canonical COCO category is hallucinated.

    `words` holds canonical categories (e.g. "person"), so a surface token is
    singularised and mapped through the thesis synonym table before comparing.
    """
    import html, re
    if not words:
        return html.escape(caption)
    bad, out = set(words), []
    sing = singularize or (lambda w: w)
    for tok in re.split(r"(\W+)", caption):
        key = inv.get(sing(tok.lower())) or inv.get(tok.lower())
        out.append(f'<span class="h">{html.escape(tok)}</span>'
                   if key in bad else html.escape(tok))
    return "".join(out)


def metrics_html(m: dict, curated: bool) -> str:
    bits = [f"CLIPScore <b>{m['clipscore']:.3f}</b>", f"{m['words']} words"]
    if curated:
        rec = "n/a" if m["recall"] is None else f"{m['recall']:.2f}"
        bits += [f"H <b>{m['H']}</b>", f"recall {rec}"]
    else:
        bits += ["no ground-truth annotations"]
    return '<div class="metrics">' + "&nbsp;&nbsp;·&nbsp;&nbsp;".join(bits) + "</div>"


class Backend:
    def __init__(self, live: bool, timeout: float, seed: int = 42):
        self.live, self.timeout, self.seed = live, timeout, seed
        self.cache = json.loads((ROOT / "demo" / "cache.json").read_text())
        self.chair, self.clip = Chair(), ClipScorer()
        self.vlms, self.prompt, self.gen = {}, None, None
        self.ready = False

    # ---------------------------------------------------------------- live
    def load(self, log=print):
        if not self.live:
            self.ready = True
            return "ready — cached mode (real thesis generations)"
        import importlib.util, torch
        from src.utils.io import load_yaml
        spec = importlib.util.spec_from_file_location(
            "demo_caption", ROOT / "scripts" / "30_demo_caption.py")
        dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)
        cfg = load_yaml(ROOT / "configs/description_eval.yaml")     # dd256 settings
        self.prompt = cfg["prompt"]["user"].strip()
        self.gen = dict(cfg["generation"])
        enc = load_yaml(ROOT / "configs/encoders.yaml")
        proj = load_yaml(ROOT / "configs/projector.yaml")
        llm = load_yaml(ROOT / "configs/llm.yaml")
        lora = load_yaml(ROOT / "configs/training_stage2.yaml").get("lora")
        for label, ck in CKPT.items():
            t0 = time.time(); log(f"[demo] loading {label} …")
            # build_vlm seeds the load (resize_token_embeddings draws the new
            # <image> row randomly) AND calls .eval(), without which LoRA
            # dropout stays on and every generation differs.
            self.vlms[label] = build_vlm(dc, ck, (enc, proj, llm, lora),
                                         seed=self.seed)
            log(f"[demo] {label} up in {time.time()-t0:.0f}s")

        warm_id = CURATED[0][0]                       # one real generation each,
        img = Image.open(image_path(warm_id)).convert("RGB")   # so the first
        for label in self.vlms:                       # live click is not slow
            t0 = time.time(); self._generate(label, img)
            log(f"[demo] {label} warm in {time.time()-t0:.1f}s")
        self.ready = True
        return f"ready — live on GPU · both models warm"

    def _generate(self, label, img):
        import torch
        with torch.no_grad():
            return self.vlms[label].generate([img], [self.prompt], **self.gen)[0].strip()

    # ------------------------------------------------------------- scoring
    def run(self, label, img, image_id):
        """-> (caption, metrics dict, source tag)"""
        curated = image_id is not None
        if self.live and self.vlms:
            t0 = time.time()
            try:
                cap = self._generate(label, img)
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"{time.time()-t0:.0f}s > {self.timeout:.0f}s")
                ch = self.chair.score(image_id, cap) if curated else {
                    "words": len(cap.split()), "H": 0, "recall": None,
                    "hallucinated": [], "n_gt": 0}
                return cap, {"caption": cap, "clipscore": self.clip.score(img, cap),
                             **{k: ch[k] for k in ("words", "H", "recall",
                                                   "hallucinated")}}, "live"
            except Exception as e:                     # never crash a live demo
                print("[demo] live failed ->", e, file=sys.stderr)
                traceback.print_exc()
                if not curated:
                    return ("Generation failed on this upload. Click one of the "
                            "ready examples below — those always work."), {
                        "caption": "", "clipscore": 0.0, "words": 0, "H": 0,
                        "recall": None, "hallucinated": []}, "error"
        if curated:
            m = dict(self.cache["entries"][str(image_id)]["models"][label])
            return m["caption"], m, "cached"
        cap = ("Cached mode has no caption for an unseen image. "
           "Click one of the ready examples below.")
        return cap, {"caption": "", "clipscore": 0.0, "words": 0, "H": 0,
                     "recall": None, "hallucinated": []}, "cached"


def build_ui(be: Backend):
    thumbs = [(str(image_path(i)), t) for i, t in CURATED]
    ids = [i for i, _ in CURATED]

    with gr.Blocks(css=CSS, title="Closing the location gap", theme=gr.themes.Soft(
            primary_hue="orange", neutral_hue="gray")) as ui:
        gr.HTML('<div id="hdr"><h1>CLOSING THE LOCATION GAP</h1>'
                '<p>Drop in any image. Same budget, same pins, one axis released.</p></div>')
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                upload = gr.Image(label="Drop an image here", type="pil", height=300,
                                  sources=["upload", "clipboard"])
                status = gr.HTML('<div id="status">loading…</div>', elem_id="status")
                gr.HTML('<div class="hint">Ready examples — always work, and carry '
                        'COCO ground truth so H and recall are shown.</div>')
                gallery = gr.Gallery(thumbs, label=None, columns=6, height=150,
                                     object_fit="cover", allow_preview=False,
                                     show_label=False)
            with gr.Column(scale=7):
                with gr.Row():
                    panels = {}
                    for label in (BASE, DRIVE):
                        with gr.Column():
                            gr.HTML(f'<div class="panel"><h3>{label}</h3></div>')
                            cap = gr.HTML('<div class="panel cap"></div>')
                            met = gr.HTML("")
                            panels[label] = (cap, met)
        gr.HTML(f'<div id="foot">{FOOTER}</div>')

        def respond(img, image_id):
            """Stream both panels; left first, then right (sequential on 11 GB)."""
            if img is None:
                return
            from PIL import ImageOps
            iid = image_id
            if iid is None:
                img = ImageOps.exif_transpose(img).convert("RGB")
                if max(img.size) > 1600:
                    img.thumbnail((1600, 1600))
            else:
                img = Image.open(image_path(iid)).convert("RGB")
            outs = {l: ('<div class="panel cap"></div>', "") for l in (BASE, DRIVE)}

            def frame(msg):
                return (outs[BASE][0], outs[BASE][1], outs[DRIVE][0], outs[DRIVE][1],
                        f'<div id="status">{msg}</div>')

            for label in (BASE, DRIVE):
                yield frame(f"generating — {label}…")
                cap, m, src = be.run(label, img, iid)
                tag = (f'<span class="tag">{src}</span>' if src != "live" else "")
                words = cap.split()
                for k in range(1, len(words) + 1):
                    shown = " ".join(words[:k])
                    body = (highlight(shown, m.get("hallucinated", []), be.chair.inv,
                                      be.chair.singularize) if iid is not None else shown)
                    outs[label] = (f'<div class="panel cap">{body}{tag}</div>', "")
                    if k % 4 == 0 or k == len(words):
                        yield frame(f"generating — {label}…")
                        time.sleep(0.012)
                outs[label] = (outs[label][0], metrics_html(m, iid is not None))
                if src == "error":
                    yield frame("generation failed — click a ready example below")
                    return
            note = "" if iid is not None else "  ·  uploaded image: no ground truth"
            yield frame("ready" + note)

        sink = [panels[BASE][0], panels[BASE][1], panels[DRIVE][0], panels[DRIVE][1],
                status]

        def pick(evt: gr.SelectData):
            return ids[evt.index]

        def from_gallery(iid):
            yield from respond(None if iid is None else "curated", iid)

        gallery.select(pick, None, state).then(
            lambda iid: None, state, upload).then(
            respond, [gr.State("curated"), state], sink)
        upload.upload(lambda: None, None, state).then(
            respond, [upload, state], sink)
        ui.load(lambda: f'<div id="status">{be.load()}</div>', None, status)
    return ui


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true")
    g.add_argument("--cached", action="store_true")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to expose on the LAN; 127.0.0.1 is\n                          enough for an SSH -L tunnel")
    a = ap.parse_args()
    be = Backend(live=a.live, timeout=a.timeout, seed=a.seed)
    build_ui(be).queue().launch(server_name=a.host, server_port=a.port,
                                share=a.share, show_api=False, quiet=False)


if __name__ == "__main__":
    main()
