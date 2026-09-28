"""Backbone registry: name -> (wrapper class, default HF repo id)."""
from __future__ import annotations

import importlib

MODELS = {
    "qwen-image-edit-2511": ("mtopsd.models.qwen_edit:QwenEditModel", "Qwen/Qwen-Image-Edit-2511"),
    "firered-image-edit-1.0": ("mtopsd.models.qwen_edit:QwenEditModel",
                               "FireRedTeam/FireRed-Image-Edit-1.0"),
    "flux2-klein-base-9b": ("mtopsd.models.flux2_klein:Flux2KleinModel",
                            "black-forest-labs/FLUX.2-klein-base-9B"),
}

# names used in the checkpoints of the original research code
ALIASES = {"qwen2511": "qwen-image-edit-2511", "firered": "firered-image-edit-1.0",
           "flux2_kleinbase": "flux2-klein-base-9b"}


def build_model(name, pretrained_path=None, device="cuda:0"):
    name = ALIASES.get(name, name)
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choices: {list(MODELS)}")
    target, hf_id = MODELS[name]
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)(pretrained_path or hf_id, device)
