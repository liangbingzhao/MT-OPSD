"""Identity-drift probe: apply e_id to a fixed image N times, save [src | r1..rN] and
return the per-round mean |r_i - src| (0-255 scale). Lower and flatter = more stable."""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def drift_eval(model, image, out_path, rounds=10, steps=30, prompt="make everything unchange",
               cfg=4.0, seed=42, panel_height=256):
    src = image.convert("RGB")
    cur, panels, drifts = src, [src], []
    for _ in range(rounds):
        cur = model.generate(prompt, cur, steps=steps, cfg=cfg, seed=seed)
        panels.append(cur)
        ref = src if cur.size == src.size else src.resize(cur.size)
        drifts.append(round(float(np.abs(np.asarray(cur.convert("RGB"), np.float32)
                                         - np.asarray(ref, np.float32)).mean()), 2))
    w, h = src.size
    pw = max(1, round(w * panel_height / h))
    strip = Image.new("RGB", (pw * len(panels), panel_height), "white")
    for i, p in enumerate(panels):
        strip.paste(p.resize((pw, panel_height)), (i * pw, 0))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    strip.save(out_path)
    return drifts
