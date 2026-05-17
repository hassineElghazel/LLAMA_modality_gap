"""Image preprocessing transforms.

Wraps standard CLIP normalization. The processor returned by HuggingFace's
``AutoProcessor`` already applies these — this module exists for cases where we
need to construct equivalent transforms outside the HF processor (e.g. inside a
custom Dataset).
"""
from __future__ import annotations

from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_eval_transform(image_size: int = 336):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
