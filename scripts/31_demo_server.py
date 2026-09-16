"""Browser interface for the captioning demo: drop an image, read the answer.

Built for the thesis defence, and deliberately dependency-free: the standard
library only, because a compute node may have no outbound pip. The model is
built by scripts/30_demo_caption.py::build, which mirrors
scripts/08_run_captioning.py::_build_vlm exactly, so what the page shows is what
the thesis measured. It is loaded ONCE at startup and stays resident.

    # on a GPU node (interactive job), then leave it running
    srun --partition=RTX --gpus=2080ti:1 --pty bash
    python scripts/31_demo_server.py \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_Cloc_long.pt

    # from your laptop, tunnel to it and open http://localhost:8080
    ssh -N -L 8080:<node>:8080 elghazel@odin

Two checkpoints can be shown side by side with --compare, which is the honest
way to demonstrate an improvement.

--dry-run serves the page without loading a model, for checking the interface.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import importlib.util
import io
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import load_yaml  # noqa: E402

MAX_UPLOAD = 20 * 1024 * 1024      # 20 MB of decoded image data


def _demo_module():
    """Import scripts/30_demo_caption.py (numeric filename -> importlib) to reuse
    its build() and caption(), so this server cannot drift from the CLI demo."""
    path = REPO_ROOT / "scripts" / "30_demo_caption.py"
    spec = importlib.util.spec_from_file_location("demo_caption", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f6;--card:#fff;--ink:#1a1a18;--mut:#6b6b66;--line:#e2e2de;--accent:#b8562f;--code:#f0efec}
@media (prefers-color-scheme:dark){:root{--bg:#17171a;--card:#202024;--ink:#ececea;--mut:#9a9a94;--line:#31313a;--accent:#e08a5f;--code:#191920}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding-block:32px;padding-left:20px;padding-right:20px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}
#drop{border:2px dashed var(--line);border-radius:10px;padding:34px 16px;text-align:center;cursor:pointer;transition:.15s}
#drop:hover,#drop.hot{border-color:var(--accent);background:var(--code)}
#drop p{margin:0;color:var(--mut)}
#prev{display:block;max-width:100%;max-height:340px;margin:14px auto 0;border-radius:8px}
label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin:14px 0 5px}
textarea,input[type=number]{width:100%;padding:9px 11px;border:1px solid var(--line);border-radius:7px;
  background:var(--bg);color:var(--ink);font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
textarea{min-height:60px;resize:vertical}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end}
.row>div{flex:1 1 150px}
button{margin-top:16px;width:100%;padding:12px;border:0;border-radius:8px;background:var(--accent);
  color:#fff;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{opacity:.55;cursor:default}
.out{white-space:pre-wrap;background:var(--code);border-radius:8px;padding:14px;margin-top:10px}
.who{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--accent);margin-bottom:6px}
.time{font-size:12px;color:var(--mut);margin-top:6px}
.err{color:#c0392b}
.note{font-size:12px;color:var(--mut);margin-top:18px}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>

<div class="card">
  <div id="drop"><p><strong>Drop an image here</strong><br>or click to choose one</p>
    <img id="prev" hidden></div>
  <input id="file" type="file" accept="image/*" hidden>
  <label for="prompt">Instruction</label>
  <textarea id="prompt">__PROMPT__</textarea>
  <div class="row"><div>
    <label for="tok">Max new tokens</label>
    <input id="tok" type="number" min="16" max="1024" step="16" value="__TOK__">
  </div></div>
  <button id="go" disabled>Describe this image</button>
</div>

<div id="res"></div>
<p class="note">__NOTE__</p>
</div><script>
const $=i=>document.getElementById(i);
let b64=null;
$('drop').onclick=()=>$('file').click();
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('hot')};
$('drop').ondragleave=()=>$('drop').classList.remove('hot');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('hot');
  if(e.dataTransfer.files[0])load(e.dataTransfer.files[0])};
$('file').onchange=e=>{if(e.target.files[0])load(e.target.files[0])};
function load(f){
  if(!f.type.startsWith('image/')){alert('That is not an image file.');return}
  const r=new FileReader();
  r.onload=()=>{b64=r.result;$('prev').src=b64;$('prev').hidden=false;$('go').disabled=false;$('res').innerHTML=''};
  r.readAsDataURL(f);
}
$('go').onclick=async()=>{
  if(!b64)return;
  $('go').disabled=true;$('go').textContent='Generating, this takes a moment...';
  $('res').innerHTML='';
  try{
    const r=await fetch('/caption',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:b64,prompt:$('prompt').value,max_new_tokens:+$('tok').value})});
    const d=await r.json();
    if(d.error){$('res').innerHTML='<div class="card err">'+esc(d.error)+'</div>'}
    else{$('res').innerHTML=d.results.map(x=>'<div class="card">'+
      (d.results.length>1?'<div class="who">'+esc(x.model)+'</div>':'')+
      '<div class="out">'+esc(x.caption)+'</div>'+
      '<div class="time">'+x.seconds.toFixed(1)+' s</div></div>').join('')}
  }catch(e){$('res').innerHTML='<div class="card err">'+esc(String(e))+'</div>'}
  $('go').disabled=false;$('go').textContent='Describe this image';
};
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    models = []          # list[(name, vlm)]
    gen_kwargs = {}
    default_prompt = ""
    lock = threading.Lock()
    dry_run = False
    page = ""

    def log_message(self, fmt, *a):      # one tidy line per request
        sys.stderr.write(f"[demo] {self.address_string()} {fmt % a}\n")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/index.html"):
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        self._send(200, self.page.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/caption":
            return self._send(404, b'{"error":"not found"}', "application/json")
        try:
            body = self._read_json()
            img_bytes = self._decode_image(body.get("image", ""))
            prompt = (body.get("prompt") or self.default_prompt).strip()
            if "<image>" not in prompt:
                raise ValueError("The instruction must contain the <image> token.")
            gen = dict(self.gen_kwargs)
            if body.get("max_new_tokens"):
                gen["max_new_tokens"] = max(16, min(1024, int(body["max_new_tokens"])))
            results = self._generate(img_bytes, prompt, gen)
            payload = json.dumps({"results": results}).encode()
        except Exception as exc:                     # never kill a live demo
            self.log_message("error: %s: %s", type(exc).__name__, exc)
            payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
        self._send(200, payload, "application/json")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD * 2:             # base64 inflates by ~4/3
            raise ValueError(f"Upload too large or empty ({n} bytes).")
        return json.loads(self.rfile.read(n))

    @staticmethod
    def _decode_image(data_url: str) -> bytes:
        if not data_url:
            raise ValueError("No image was uploaded.")
        raw = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
        try:
            blob = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("The upload was not valid base64 image data.")
        if len(blob) > MAX_UPLOAD:
            raise ValueError(f"Image is {len(blob)//1024//1024} MB; the limit is "
                             f"{MAX_UPLOAD//1024//1024} MB.")
        return blob

    def _generate(self, img_bytes: bytes, prompt: str, gen: dict) -> list[dict]:
        if self.dry_run:
            time.sleep(0.4)
            return [{"model": "dry-run", "seconds": 0.4,
                     "caption": f"[dry run] {len(img_bytes)} bytes, "
                                f"max_new_tokens={gen.get('max_new_tokens')}, "
                                f"prompt={prompt!r}"}]
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        out = []
        # One GPU: serialise generation rather than letting two requests collide.
        with self.lock:
            for name, vlm in self.models:
                t0 = time.time()
                text = vlm.generate([img], [prompt], **gen)[0].strip()
                out.append({"model": name, "caption": text,
                            "seconds": round(time.time() - t0, 2)})
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vlm-checkpoint", default="outputs/checkpoints/stage2_vlm_Cloc_long.pt")
    p.add_argument("--compare", default=None, help="second checkpoint, shown side by side")
    p.add_argument("--config", default="configs/description_eval.yaml",
                   help="prompt + generation settings; the default is the 256-token "
                        "detail instruction the models were trained on")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 (default) is reachable only through an SSH tunnel")
    p.add_argument("--dry-run", action="store_true",
                   help="serve the page without loading a model")
    args = p.parse_args()

    cap_cfg = load_yaml(args.config)
    prompt = cap_cfg["prompt"]["user"].strip()
    gen = dict(cap_cfg["generation"])
    if args.max_new_tokens:
        gen["max_new_tokens"] = args.max_new_tokens

    models = []
    if not args.dry_run:
        demo = _demo_module()
        enc_cfg = load_yaml(args.encoders_config)
        proj_cfg = load_yaml(args.projector_config)
        llm_cfg = load_yaml(args.llm_config)
        lora_cfg = load_yaml(args.stage2_config).get("lora")
        for ck in [args.vlm_checkpoint] + ([args.compare] if args.compare else []):
            if str(ck).lower() != "random" and not Path(ck).exists():
                raise SystemExit(f"[demo] checkpoint not found: {ck}")
            print(f"[demo] loading {ck} ...", flush=True)
            t0 = time.time()
            models.append((Path(ck).stem, demo.build(ck, enc_cfg, proj_cfg, llm_cfg, lora_cfg)))
            print(f"[demo]   ready in {time.time()-t0:.0f}s", flush=True)

    names = ", ".join(n for n, _ in models) or "dry run, no model loaded"
    Handler.models = models
    Handler.gen_kwargs = gen
    Handler.default_prompt = prompt
    Handler.dry_run = args.dry_run
    Handler.page = (PAGE
        .replace("__TITLE__", "Image captioning demo")
        .replace("__SUB__", f"{names} &middot; {gen.get('max_new_tokens')} tokens max")
        .replace("__PROMPT__", prompt.replace("&", "&amp;").replace("<", "&lt;"))
        .replace("__TOK__", str(gen.get("max_new_tokens", 256)))
        .replace("__NOTE__", "Generation is deterministic (greedy, no sampling). "
                             "This model hallucinates an object in roughly four of "
                             "five detailed descriptions, so read the output critically."))

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    node = socket.gethostname()
    print(f"\n[demo] serving on http://{args.host}:{args.port}  (node {node})")
    if args.host == "127.0.0.1":
        print(f"[demo] from your laptop:  ssh -N -L {args.port}:{node}:{args.port} "
              f"{__import__('getpass').getuser()}@odin")
        print(f"[demo] then open:         http://localhost:{args.port}")
    else:
        print("[demo] WARNING: bound to a non-loopback address; anyone who can reach "
              "this node can use it.")
    print("[demo] Ctrl-C to stop.\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopped.")
        srv.shutdown()


if __name__ == "__main__":
    main()
