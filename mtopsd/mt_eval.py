"""In-training multi-turn metric on sentinel sessions (assets/mt_sentinels).

Each sentinel session is run twice from the same source image:
  A) plain  - every turn edits the previous plain output (normal inference);
  B) emu    - after every turn, Emu Edit sequential thresholding (arXiv 2311.10089, Sec. 4.4)
              reverts pixels the edit barely changed back to the previous image, and the
              cleaned image is fed forward.
The logged drift at turn t is mean |A_t - B_t| (0-255). A stable model adds little change
outside the region an instruction targets, so the two chains stay close; a model that
accumulates noise drifts away from its thresholded self. Unlike |A_t - source|, legitimate
edits do not count toward it. The per-turn colorfulness of chain A is logged too: the
rainbow-noise collapse of these backbones shows up as a colorfulness spike.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_tensor(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)          # 1,3,H,W in [0,1]


def _gaussian_blur(d, sigma):
    """Separable Gaussian low-pass on a [1,1,H,W] map (replicate padding)."""
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).to(d)
    pad = ksize // 2
    d = F.conv2d(F.pad(d, (pad, pad, 0, 0), mode="replicate"), k.view(1, 1, 1, ksize))
    return F.conv2d(F.pad(d, (0, 0, pad, pad), mode="replicate"), k.view(1, 1, ksize, 1))


def emu_threshold(prev_pil, raw_pil, alpha=0.03, blur_sigma=3.0):
    """Keep pixels of `raw` whose low-passed mean-RGB change from `prev` is >= alpha (images in
    [0, 1]); revert the rest to `prev`."""
    if raw_pil.size != prev_pil.size:
        raw_pil = raw_pil.resize(prev_pil.size, Image.BICUBIC)
    p, r = _to_tensor(prev_pil), _to_tensor(raw_pil)
    d_bar = _gaussian_blur((r - p).abs().mean(dim=1, keepdim=True), blur_sigma)
    mask = (d_bar >= float(alpha)).float()
    out = mask * r + (1.0 - mask) * p
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255.0
    return Image.fromarray(out.astype(np.uint8), "RGB")


def colorfulness(pil):
    """Hasler-Suesstrunk colorfulness."""
    a = np.asarray(pil.convert("RGB")).astype(np.float32)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def load_sentinels(path):
    """[(id, source PIL, instructions)] from a sentinels json (images/<id>.png next to it)."""
    with open(path) as f:
        items = json.load(f)["items"]
    img_dir = os.path.join(os.path.dirname(path), "images")
    return [(it["id"], Image.open(os.path.join(img_dir, f"{it['id']}.png")).convert("RGB"),
             it["turns"]) for it in items]


def _strip(panels):
    w, h = panels[0].size
    strip = Image.new("RGB", (w * len(panels), h), "white")
    for i, p in enumerate(panels):
        strip.paste(p.resize((w, h)), (i * w, 0))
    return strip


@torch.no_grad()
def plain_vs_emu(model, src, instructions, out_dir, steps=30, cfg=4.0, seed=42,
                 alpha=0.03, blur_sigma=3.0):
    """Returns (per-turn drift |A_t - B_t|, per-turn colorfulness of A); saves both strips."""
    os.makedirs(out_dir, exist_ok=True)
    src = src.convert("RGB")
    a_cur, b_cur = src, src
    plains, emus, drifts, cfs = [], [], [], []
    for t, instr in enumerate(instructions):
        a_raw = model.generate(instr, a_cur, steps=steps, cfg=cfg, seed=seed)
        # turn 1 starts both chains from the same image and seed: reuse A's output
        b_raw = a_raw if t == 0 else model.generate(instr, b_cur, steps=steps, cfg=cfg, seed=seed)
        b_prev = b_cur if b_cur.size == b_raw.size else b_cur.resize(b_raw.size, Image.BICUBIC)
        b_clean = emu_threshold(b_prev, b_raw, alpha, blur_sigma)
        b_cmp = b_clean if b_clean.size == a_raw.size else b_clean.resize(a_raw.size, Image.BICUBIC)
        drifts.append(round(float(np.abs(np.asarray(a_raw, np.float32)
                                         - np.asarray(b_cmp, np.float32)).mean()), 2))
        cfs.append(round(colorfulness(a_raw), 1))
        plains.append(a_raw)
        emus.append(b_clean)
        a_cur, b_cur = a_raw, b_clean
    _strip([src] + plains).save(os.path.join(out_dir, "strip_plain.png"))
    _strip([src] + emus).save(os.path.join(out_dir, "strip_emu.png"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"drift_plain_vs_emu": drifts, "colorfulness_plain": cfs,
                   "instructions": instructions, "steps": steps, "cfg": cfg, "seed": seed,
                   "emu": {"alpha": alpha, "blur_sigma": blur_sigma}}, f, indent=1)
    return drifts, cfs
