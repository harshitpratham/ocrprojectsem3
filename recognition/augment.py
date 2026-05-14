"""
Handwriting-oriented image augmentations for TrOCR v2 domain adaptation.

Used by train_v2_domain_adapt.py and can be composed into torchvision / PIL pipelines.
"""

from __future__ import annotations

import random
from typing import Tuple

from PIL import Image, ImageEnhance, ImageFilter


def elastic_warp_pil(img: Image.Image, alpha: float = 8.0, sigma: float = 3.0) -> Image.Image:
    """
    Lightweight elastic-style distortion using sequential small affine jitter.
    Full elastic grid is expensive; this approximates ink wobble for student handwriting.
    """
    if random.random() > 0.5:
        img = img.rotate(random.uniform(-4, 4), expand=True, fillcolor=255)
        w, h = img.size
        img = img.crop(((w // 8), (h // 8), w - w // 8, h - h // 8))
        img = img.resize((w, h), Image.Resampling.BILINEAR)
    return img


def ink_fade(img: Image.Image) -> Image.Image:
    """Simulate faded pen / uneven pressure."""
    if random.random() > 0.4:
        factor = random.uniform(0.55, 1.0)
        img = ImageEnhance.Brightness(img).enhance(factor)
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.15))
    return img


def paper_noise(img: Image.Image) -> Image.Image:
    """JPEG-style compression + light blur for phone-captured field sheets."""
    if random.random() > 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.6)))
    if random.random() > 0.25:
        q = random.randint(55, 90)
        import io

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def augment_word_crop(img: Image.Image, *, aggressive: bool = False) -> Image.Image:
    """Apply v2 augmentation stack to a single RGB PIL image."""
    img = img.convert("RGB")
    img = elastic_warp_pil(img, alpha=12.0 if aggressive else 8.0, sigma=3.0)
    img = ink_fade(img)
    if aggressive:
        img = paper_noise(img)
    return img


def jitter_box(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_frac: float = 0.04,
) -> Tuple[int, int, int, int]:
    """Randomly expand/shrink crop by up to pad_frac of min(image side)."""
    x1, y1, x2, y2 = box
    pad = int(min(img_w, img_h) * pad_frac)
    dx1 = random.randint(-pad, pad)
    dy1 = random.randint(-pad, pad)
    dx2 = random.randint(-pad, pad)
    dy2 = random.randint(-pad, pad)
    nx1 = max(0, min(x1 + dx1, img_w - 1))
    ny1 = max(0, min(y1 + dy1, img_h - 1))
    nx2 = max(nx1 + 1, min(x2 + dx2, img_w))
    ny2 = max(ny1 + 1, min(y2 + dy2, img_h))
    return nx1, ny1, nx2, ny2
