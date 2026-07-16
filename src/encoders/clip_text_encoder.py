"""Frozen CLIP text tower + frozen linear lift into LLaMA's 4096-d space.

The vision side (``clip_encoder.py``) uses ``openai/clip-vit-large-patch14`` with
"no projection head and no text tower". This module adds the MATCHING text tower
of the SAME CLIP checkpoint so we can build a *semantically-grounded* alignment
anchor (CLIP text was contrastively trained to sit near its image), replacing the
flat LLaMA lookup-embedding anchor used until now.

Two-stage, both FROZEN:
    caption --CLIPTokenizer--> CLIPTextModelWithProjection --> text_embeds (768-d,
        CLIP's image-aligned space)
    --W--> (4096-d)   with W a FIXED seeded semi-orthogonal 768->4096 lift.

Why a semi-orthogonal W (not Gaussian, not learned): CLIP only ever learned a
768-d text projection; reaching LLaMA's 4096-d needs parameters CLIP never
trained, so the extra dimensions are unavoidably non-CLIP. A seeded semi-
orthogonal map (orthonormal columns, ``W^T W = I_768``) is norm- and angle-
preserving (``||W x|| = ||x||``, ``(W x)·(W y) = x·y``), so it transplants CLIP's
relative geometry into 4096-d essentially intact (Johnson-Lindenstrauss). It is
NOT trained, so it adds zero learnable capacity to the comparison, and it is
reproducible from the seed alone (no checkpoint to ship).

This anchor is a LOSS TARGET only (Cloc's ``mu_y`` / Corient's InfoNCE positive).
It is NEVER fed into the LLM: the decoder keeps reading LLaMA embeddings for the
prompt and the generated caption. Only the connector is pulled toward this anchor.

IDENTICAL FROZEN ANCHOR ACROSS JOBS. The comparison is only valid if Cloc and
Corient are pulled toward the SAME CLIP geometry. Two guarantees enforce that:
  1) CLIP text weights come from a fixed pretrained checkpoint (``hf_id``) -> byte
     identical wherever loaded.
  2) The lift ``W`` is MATERIALISED ONCE to ``lift_path`` and every subsequent job
     LOADS that exact file (``.load()`` builds-and-saves only if it is absent).
     Regeneration from the seed is deterministic too, but persisting removes any
     dependence on torch-version / RNG-placement drift. Cloc's ``mu_y`` is built
     with this same W (so it is frozen into the saved centroid), and Corient loads
     the same W for its live InfoNCE positive -> both anchor to one geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn


@dataclass
class CLIPTextEncoderConfig:
    hf_id: str = "openai/clip-vit-large-patch14"
    device: str = "cuda"
    weights_dtype: str = "bfloat16"
    max_length: int = 77            # CLIP text context length
    out_dim: int = 4096             # LLaMA-2 input embedding dim
    proj_seed: int = 42             # seed for the fixed lift
    lift_method: str = "semi_orthogonal"   # "semi_orthogonal" (recommended) | "random"
    # Canonical on-disk artifact for the lift. Built once, then loaded by every job
    # so all training runs share a byte-identical W. None => always regenerate from
    # seed (still deterministic, but no cross-job persistence guarantee).
    lift_path: Optional[str] = "outputs/anchors/clip_lift.pt"


def _semi_orthogonal(out_dim: int, in_dim: int, seed: int) -> torch.Tensor:
    """Fixed (out_dim, in_dim) matrix with ORTHONORMAL COLUMNS (out_dim > in_dim),
    generated deterministically from ``seed`` via a sign-canonicalised reduced QR.

    ``W^T W = I_{in_dim}`` -> ``||W x|| = ||x||`` and ``(W x)·(W y) = x·y`` for all
    x, y, so the CLIP text geometry survives the lift into 4096-d unchanged. QR
    (rather than ``nn.init.orthogonal_``) keeps the result identical across torch
    versions that differ on the ``generator=`` kwarg.
    """
    if out_dim < in_dim:
        raise ValueError(f"need out_dim >= in_dim for orthonormal columns, got {out_dim} < {in_dim}")
    g = torch.Generator().manual_seed(int(seed))
    a = torch.randn(out_dim, in_dim, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a, mode="reduced")            # q: (out_dim, in_dim), r: (in_dim, in_dim)
    # Canonical signs: make diag(r) positive so q is uniquely determined by `a`.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(torch.float32)                           # (out_dim, in_dim)


def _random_lift(out_dim: int, in_dim: int, seed: int) -> torch.Tensor:
    """Fixed (out_dim, in_dim) Gaussian lift, seeded, scaled by 1/sqrt(in_dim) so
    output variance ~ input variance. Supported for ablation; NOT norm/angle
    preserving (semi-orthogonal is preferred). Generated on CPU for device-
    independent reproducibility."""
    g = torch.Generator().manual_seed(int(seed))
    a = torch.randn(out_dim, in_dim, generator=g, dtype=torch.float64)
    return (a / (in_dim ** 0.5)).to(torch.float32)


def _build_lift(method: str, out_dim: int, in_dim: int, seed: int) -> torch.Tensor:
    if method == "semi_orthogonal":
        return _semi_orthogonal(out_dim, in_dim, seed)
    if method == "random":
        return _random_lift(out_dim, in_dim, seed)
    raise ValueError(f"lift_method must be 'semi_orthogonal' or 'random', got {method!r}")


class CLIPTextTower(nn.Module):
    """Frozen CLIP text encoder returning a 4096-d anchor embedding per caption.

    All parameters (text transformer, CLIP text_projection, and the lift ``W``)
    are frozen. ``encode`` runs under ``no_grad`` and returns fp32 on the
    configured device, matching how the trainers compute geometry terms in fp32.
    """

    def __init__(self, cfg: Optional[CLIPTextEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or CLIPTextEncoderConfig()
        self._text = None
        self._tokenizer = None
        # 768->4096 lift as a frozen buffer (moves with .to(device), never trained,
        # never in an optimizer). in_dim filled at load() from the model's projection.
        self.register_buffer("W", torch.empty(0), persistent=False)

    def load(self) -> "CLIPTextTower":
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        dtype = getattr(torch, self.cfg.weights_dtype)
        self._text = (
            CLIPTextModelWithProjection.from_pretrained(self.cfg.hf_id, torch_dtype=dtype)
            .to(self.cfg.device)
            .eval()
        )
        for p in self._text.parameters():
            p.requires_grad = False
        self._tokenizer = CLIPTokenizer.from_pretrained(self.cfg.hf_id)
        in_dim = int(self._text.config.projection_dim)   # 768 for ViT-L/14
        self.W = self._load_or_build_lift(in_dim).to(self.cfg.device)
        return self

    def _load_or_build_lift(self, in_dim: int) -> torch.Tensor:
        """Return the frozen lift W, loading the canonical on-disk artifact if it
        exists so every job shares a byte-identical matrix; otherwise build it from
        (method, seed) and persist it to ``lift_path`` with a metadata sidecar."""
        want = (self.cfg.out_dim, in_dim)
        path = Path(self.cfg.lift_path) if self.cfg.lift_path else None
        if path is not None and path.exists():
            blob = torch.load(str(path), map_location="cpu")
            W = blob["W"] if isinstance(blob, dict) else blob
            if tuple(W.shape) != want:
                raise ValueError(
                    f"lift at {path} has shape {tuple(W.shape)}, expected {want}. "
                    f"Delete it to rebuild, or point lift_path elsewhere."
                )
            if isinstance(blob, dict):
                meta = {k: blob.get(k) for k in ("method", "seed")}
                if meta["method"] != self.cfg.lift_method or meta["seed"] != self.cfg.proj_seed:
                    raise ValueError(
                        f"lift at {path} was built with {meta}, but config asks for "
                        f"method={self.cfg.lift_method!r} seed={self.cfg.proj_seed}. "
                        f"Reuse the existing artifact or point lift_path elsewhere."
                    )
            return W.to(torch.float32)
        W = _build_lift(self.cfg.lift_method, self.cfg.out_dim, in_dim, self.cfg.proj_seed)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"W": W, "method": self.cfg.lift_method, "seed": int(self.cfg.proj_seed),
                 "hf_id": self.cfg.hf_id, "in_dim": in_dim, "out_dim": self.cfg.out_dim},
                str(path),
            )
        return W

    def _require_loaded(self) -> None:
        if self._text is None:
            raise RuntimeError("CLIPTextTower not loaded — call .load() first")

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim

    @torch.no_grad()
    def encode(self, captions: list[str], batch_size: int = 256) -> torch.Tensor:
        """Return (N, 4096) fp32 anchor embeddings for a list of captions.

        text_embeds (CLIP's image-aligned 768-d, post text_projection) lifted by
        the fixed semi-orthogonal W into 4096-d. Not L2-normalised here: InfoNCE
        normalises internally, and Cloc's distance term uses the raw centroid with
        a frozen ``trace_x`` normaliser (matching the LLaMA-anchor convention).
        """
        self._require_loaded()
        device = self.cfg.device
        out: list[torch.Tensor] = []
        for i in range(0, len(captions), batch_size):
            chunk = captions[i : i + batch_size]
            enc = self._tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=self.cfg.max_length,
            ).to(device)
            emb = self._text(**enc).text_embeds            # (b, 768), model dtype
            lifted = emb.to(torch.float32) @ self.W.t()    # (b, 4096) fp32
            out.append(lifted)
        return torch.cat(out, dim=0)


def build_clip_text_tower(
    encoders_cfg: dict | None = None,
    *,
    out_dim: int = 4096,
    proj_seed: int = 42,
    lift_method: str = "semi_orthogonal",
    lift_path: Optional[str] = "outputs/anchors/clip_lift.pt",
) -> CLIPTextTower:
    """Construct a CLIPTextTower, reusing hf_id/device/dtype from configs/encoders.yaml
    so the text tower always matches the vision tower's CLIP checkpoint. ``lift_path``
    is the shared on-disk lift artifact — pass the SAME value in every job so all
    runs load one byte-identical W."""
    if encoders_cfg is None:
        return CLIPTextTower(CLIPTextEncoderConfig(
            out_dim=out_dim, proj_seed=proj_seed,
            lift_method=lift_method, lift_path=lift_path,
        ))
    vm = encoders_cfg.get("vision_model", {})
    inf = encoders_cfg.get("inference", {})
    cfg = CLIPTextEncoderConfig(
        hf_id=vm.get("hf_id", "openai/clip-vit-large-patch14"),
        device=inf.get("device", "cuda"),
        weights_dtype=inf.get("weights_dtype", "bfloat16"),
        out_dim=out_dim,
        proj_seed=proj_seed,
        lift_method=lift_method,
        lift_path=lift_path,
    )
    return CLIPTextTower(cfg)
