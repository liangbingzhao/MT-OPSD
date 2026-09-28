"""Load a base model or an MT-OPSD checkpoint and run multi-turn editing sessions."""
from __future__ import annotations

import json
import os

import torch
from PIL import Image

from .models import build_model

DEFAULTS = {"train_area": 512 * 512, "steps": 30, "cfg": 4.0}


def read_checkpoint_config(ckpt_dir):
    """LoRA/inference settings of a checkpoint dir: adapter_config.json, else the args stored
    in training_state.pt (checkpoints of the original research code)."""
    p = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    ts = os.path.join(ckpt_dir, "training_state.pt")
    if not os.path.isfile(ts):
        raise FileNotFoundError(f"{ckpt_dir} has neither adapter_config.json nor training_state.pt")
    a = torch.load(ts, map_location="cpu", weights_only=False)["args"]
    return {"model": a["model"], "rank": a["rank"], "lora_alpha": a["lora_alpha"],
            "target_modules": list(a["target_modules"]), "train_area": a.get("train_area"),
            "steps": a.get("rollout_steps"), "cfg": a.get("rollout_cfg")}


def load_model(ckpt=None, model=None, pretrained_path=None, device="cuda:0", train_area=None):
    """Returns (model wrapper, settings). With `ckpt`, the backbone and LoRA settings come from
    the checkpoint; `model` names the backbone for base-model runs."""
    cfg = dict(DEFAULTS)
    if ckpt:
        cfg.update({k: v for k, v in read_checkpoint_config(ckpt).items() if v is not None})
    elif model:
        cfg["model"] = model
    else:
        raise ValueError("pass a checkpoint dir or a base model name")
    if train_area:
        cfg["train_area"] = train_area
    m = build_model(cfg["model"], pretrained_path, device)
    m.set_resolution(cfg["train_area"])
    if ckpt:
        m.load_lora(ckpt, cfg["rank"], cfg["lora_alpha"], cfg["target_modules"])
    return m, cfg


@torch.no_grad()
def edit_session(model, image, instructions, steps=30, cfg=4.0, seed=42, save_fn=None):
    """Apply the instructions turn by turn, each turn editing the previous output."""
    cur, outs = image.convert("RGB"), []
    for t, instruction in enumerate(instructions):
        cur = model.generate(instruction, cur, steps=steps, cfg=cfg, seed=seed)
        if save_fn is not None:
            save_fn(t, cur)
        outs.append(cur)
    return outs


def _valid_image(path):
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001  (missing or truncated)
        return False


def run_sessions(items, images_dir, out_dir, ckpt=None, model=None, pretrained_path=None,
                 device="cuda:0", train_area=None, steps=None, cfg=None, seed=42, tag="sample"):
    """Generate benchmark sessions into <out_dir>/<id>/{source,turn1..turnN}.png.

    items: [{"id", "turns": [...]}]; the source image is <images_dir>/<id>.png. Resumable:
    finished turns are skipped, and since every turn uses the same fixed seed, continuing
    from the saved previous turn is identical to an uninterrupted run. The model is only
    loaded if something is left to generate."""
    todo = []
    for it in items:
        d = os.path.join(out_dir, it["id"])
        start = next((t for t in range(len(it["turns"]))
                      if not _valid_image(os.path.join(d, f"turn{t + 1}.png"))), len(it["turns"]))
        if start < len(it["turns"]):
            todo.append((it, start))
    print(f"[{tag}] {len(items) - len(todo)}/{len(items)} sessions already generated", flush=True)
    if not todo:
        return
    m, settings = load_model(ckpt, model, pretrained_path, device, train_area)
    steps = steps or settings["steps"]
    cfg = cfg if cfg is not None else settings["cfg"]
    for n, (it, start) in enumerate(todo, 1):
        d = os.path.join(out_dir, it["id"])
        os.makedirs(d, exist_ok=True)
        src = Image.open(os.path.join(images_dir, f"{it['id']}.png")).convert("RGB")
        src.save(os.path.join(d, "source.png"))
        cur = src if start == 0 else Image.open(os.path.join(d, f"turn{start}.png")).convert("RGB")
        for t in range(start, len(it["turns"])):
            cur = m.generate(it["turns"][t], cur, steps=steps, cfg=cfg, seed=seed)
            cur.save(os.path.join(d, f"turn{t + 1}.png"))
        print(f"[{tag}] {n}/{len(todo)} {it['id']}", flush=True)
