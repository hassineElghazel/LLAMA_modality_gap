"""Stage 1: projector-only pretraining (modality substitution).

LLM frozen, encoder frozen, only projector params get gradients
(~17M parameters). Feasible on 16GB GPU because no LLM gradient memory is
needed.

Per the plan §8.2 B.1, the recipe follows the ReVision modality-substitution
scheme over Bunny-pretrain 1M. Implementation skeleton — fill in the loss
formulation when integrating with the reference repo (see
`references/Yu-xm/ReVision/`).
"""
from __future__ import annotations

from pathlib import Path

import torch
from rich.console import Console

from ..models.checkpoint import save_projector
from ..models.projector import MLP2xGELU
from .trainer_utils import build_adamw, cosine_with_warmup, freeze_module

console = Console()


def train_stage1(
    projector: MLP2xGELU,
    llm: torch.nn.Module,
    encoder,
    text_dataloader,
    cfg: dict,
) -> Path:
    """Train projector with LLM and encoder frozen.

    Loss: language-modeling loss on text-only modality-substitution inputs
    (per ReVision). Concrete loss assembly is delegated to caller — this
    function provides the optimization loop, not the data formulation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freeze_module(llm)
    freeze_module(encoder)
    projector.to(device).train()

    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["schedule"]
    total_steps = len(text_dataloader) * sched_cfg["num_epochs"]
    warmup_steps = int(total_steps * sched_cfg["warmup_ratio"])

    optimizer = build_adamw(
        projector.parameters(),
        lr=opt_cfg["lr"],
        wd=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
    )
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    log_every = cfg["logging"]["log_every_steps"]
    save_every = cfg["logging"]["save_every_steps"]
    ckpt_path = Path(cfg["output"]["checkpoint_path"])

    step = 0
    for epoch in range(sched_cfg["num_epochs"]):
        for batch in text_dataloader:
            optimizer.zero_grad(set_to_none=True)
            # NOTE: caller-provided collator must produce {"loss": <scalar>}
            # via the modality-substitution forward. Placeholder forward:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = batch["forward_fn"]()  # contract: returns object with .loss
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % log_every == 0:
                console.log(f"[stage1] epoch={epoch} step={step} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")
            if save_every and step % save_every == 0:
                save_projector(projector, ckpt_path)

    save_projector(projector, ckpt_path)
    return ckpt_path
