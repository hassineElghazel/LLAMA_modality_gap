"""Project the image and text clouds into a shared 3D frame for the gap figure.

Runs on CPU on a login node: it loads the pooled embeddings, builds ONE frame
from the baseline condition, and writes a small JSON of coordinates. Nothing
here touches a GPU or the running job.

THE FRAME.  The origin is the text centroid, which is the same for every
condition because the text tower is frozen. The first axis is the baseline gap
direction u = (xbar_base - ybar)/||.||, so a point's first coordinate is its
position along the axis the method drives. The other two are the leading
directions of the baseline image cloud's variance orthogonal to u. The frame is
fitted ONCE, on the baseline, and applied unchanged to every condition -- a
picture where each panel gets its own projection can show anything.

HONESTY FIELDS.  For each condition the output also carries, in full 4096-d:
``G_mu`` (the true centroid distance), ``along_u`` and ``perp`` (how that
displacement splits inside and outside the frame's first axis), and
``var_captured`` (the share of the image cloud's variance the three axes hold).
A reader can check the picture against those rather than trusting the geometry
of a projection.

Usage:
    python scripts/33_gap_3d.py \
        --conditions C3pinr C5bp_lam0p01 C5bp_lam0p1 Clocorient \
        --base C3pinr --out outputs/metrics/gap_3d.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_pair(emb_dir: Path, tag: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.load(emb_dir / f"projected_{tag}_image_pooled.pt", map_location="cpu")
    y = torch.load(emb_dir / f"projected_{tag}_text_pooled.pt", map_location="cpu")
    return x.float(), y.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--base", required=True, help="condition whose geometry fixes the frame")
    p.add_argument("--embeddings-dir", default="outputs/embeddings")
    p.add_argument("--out", default="outputs/metrics/gap_3d.json")
    p.add_argument("--points", type=int, default=600, help="points kept per cloud")
    p.add_argument("--frame", choices=["variance", "displacement"], default="displacement",
                   help="what axes 2-3 span. 'displacement' (default) uses the "
                        "leading directions of the OTHER conditions' centroid "
                        "offsets perpendicular to u, so every centroid is placed "
                        "at close to its true distance from the text cloud. "
                        "'variance' uses image-cloud variance instead, which "
                        "draws cloud shape better but can place centroids far "
                        "short of their real G_mu.")
    p.add_argument("--pc-source", choices=["base", "union"], default="union",
                   help="whose variance sets axes 2 and 3. 'union' pools every "
                        "condition so no single cloud is favoured; 'base' uses "
                        "the baseline alone. The gap axis is the baseline's "
                        "either way, and the frame is still fitted only once.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    emb = Path(args.embeddings_dir)
    g = torch.Generator().manual_seed(args.seed)

    # ----- frame, from the baseline only ------------------------------------
    Xb, Yb = load_pair(emb, args.base)
    ybar = Yb.mean(0)
    u = Xb.mean(0) - ybar
    gmu_base = float(u.norm())
    u = u / u.norm()

    if args.frame == "displacement":
        # Axes 2-3 span where the centroids actually went. Closure is not a
        # translation down u: measured on this run, u's share of the offset falls
        # to 6% by lambda_d=0.1, so a variance frame would draw the clouds much
        # closer to the text than they are.
        offs = []
        for t_ in args.conditions:
            if t_ == args.base:
                continue
            Xc, Yc = load_pair(emb, t_)
            dd = Xc.mean(0) - Yc.mean(0)
            offs.append(dd - (dd @ u) * u)     # ... perpendicular part only
            del Xc, Yc
        Dm = torch.stack(offs)                                  # (C-1, 4096)
        V = torch.linalg.svd(Dm, full_matrices=False).Vh.T
        nrm = Dm.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cos = (Dm / nrm) @ (Dm / nrm).T
        print("[3d] cosines between perpendicular offsets:\n" +
              "\n".join("     " + "  ".join(f"{v:6.3f}" for v in r) for r in cos.tolist()))
    else:
        # Axes 2-3 span image variance orthogonal to u. Pooling every condition
        # keeps the frame from flattering the baseline.
        src = [Xb] if args.pc_source == "base" else \
              [Xb] + [load_pair(emb, t_)[0] for t_ in args.conditions if t_ != args.base]
        R = torch.cat([S_ - S_.mean(0) for S_ in src], dim=0)
        R = R - (R @ u).unsqueeze(1) * u
        _, _, V = torch.pca_lowrank(R, q=6, center=False)
        del R, src
    basis = torch.stack([u, V[:, 0], V[:, 1]], dim=1)          # (4096, 3)
    del Xb

    orth = (basis.T @ basis - torch.eye(3)).abs().max()
    print(f"[3d] frame from {args.base}: G_mu={gmu_base:.3f}, "
          f"orthonormality error {float(orth):.2e}")

    out = {"base": args.base, "n_points": args.points,
           "axes": ["gap direction u (baseline)", "image PC1 orthogonal to u",
                    "image PC2 orthogonal to u"],
           "origin": f"text centroid of {args.base}", "conditions": []}

    for tag in args.conditions:
        X, Y = load_pair(emb, tag)
        d = X.mean(0) - Y.mean(0)
        along = float(d @ u)
        gmu = float(d.norm())
        perp = float((d - along * u).norm())

        Xc = X - X.mean(0)
        tot = float((Xc ** 2).sum())
        cap = float(((Xc @ basis) ** 2).sum())

        def take(T):
            idx = torch.randperm(T.shape[0], generator=g)[: args.points]
            return ((T[idx] - ybar) @ basis).tolist()

        out["conditions"].append({
            "tag": tag, "n_image": int(X.shape[0]), "n_text": int(Y.shape[0]),
            "G_mu": gmu, "along_u": along, "perp": perp,
            "var_captured": cap / tot,
            "trace_image": tot / (X.shape[0] - 1),
            "trace_text": float(((Y - Y.mean(0)) ** 2).sum()) / (Y.shape[0] - 1),
            "centroid_in_frame": float(((X.mean(0) - Y.mean(0)) @ basis).norm()),
            "image_centroid": ((X.mean(0) - ybar) @ basis).tolist(),
            "text_centroid": ((Y.mean(0) - ybar) @ basis).tolist(),
            "image_xyz": take(X), "text_xyz": take(Y),
        })
        c3 = float(((X.mean(0) - Y.mean(0)) @ basis).norm())
        print(f"[3d] {tag:14s} G_mu={gmu:8.3f}  along_u={along:8.3f}  perp={perp:7.3f}  "
              f"|centroid in frame|={c3:8.3f} ({100*c3/gmu:5.1f}% of G_mu)  "
              f"var_captured={100*cap/tot:4.1f}%")
        del X, Y, Xc

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"[3d] wrote {args.out} ({Path(args.out).stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
