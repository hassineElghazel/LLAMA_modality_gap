# Third-party code attributions

This file tracks any code vendored from external repositories. Per the plan §5,
the strategy is "vendored reference, clean reimplementation" by default, with
selective vendoring (Strategy B) for ReAlign / AnisoAlign algorithm scripts so
that comparison numbers cannot be confounded by reimplementation differences.

Each entry below lists: source repo, commit hash at time of copy, license, and
where the vendored code lives in this repo.

---

## Vendored code

(none yet — populate as files are copied from `references/`)

| Path in this repo | Source repo | Commit | License | Notes |
|---|---|---|---|---|
| `configs/deepspeed/zero2.json` | `Yu-xm/ReVision` | `9353aea` | (see ReVision LICENSE) | Verbatim copy of `script/deepspeed/zero2.json`. Used by Stage 1 + Stage 2 trainers. |
| `configs/deepspeed/zero3.json` | `Yu-xm/ReVision` | `9353aea` | (see ReVision LICENSE) | Verbatim copy of `script/deepspeed/zero3.json`. Reserved for memory-tight Stage 2 runs. |

---

## Reference repos consulted (not vendored)

These repos are cloned into `references/` (gitignored) for reading and
verification only. Their code is reimplemented in `src/` from scratch.

| Repo | Purpose |
|---|---|
| `Yu-xm/ReVision` | ReAlign / ReVision (Feb 2026) — algorithm reference + env setup |
| `Yu-xm/Modality_Gap_Theory` | AnisoAlign (May 2026) — diagnostic metrics + figure plotting |
| `Yu-xm/Unicorn` | Unicorn (Mar 2025) — LLaVA-style projector + Stage 1 trainer |
| `Weixin-Liang/Modality-Gap` | Mind the Gap (2022) — independent cross-check |

Before vendoring anything from these repos, verify license is permissive (MIT,
Apache 2.0, BSD). Flag any GPL / non-commercial license.
