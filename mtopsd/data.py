"""OmniEdit-Filtered-1.2M source-image / instruction stream.

MT-OPSD needs no target images: each step only uses a clean source image I_0 and an
editing instruction e. The dataset reads the parquet shards of
TIGER-Lab/OmniEdit-Filtered-1.2M directly with pyarrow (zero-copy, lock-free, so
several jobs can share one copy) and samples task-balanced: pick one of the 6 tasks
uniformly, then a random example of that task.
"""
from __future__ import annotations

import bisect
import glob
import io
import os
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ParquetRows:
    """Random row access over parquet shards, reading one row group at a time."""

    COLUMNS = ["task", "src_img", "edited_prompt_list", "omni_edit_id"]

    def __init__(self, files):
        import pyarrow.parquet as pq

        self.files = list(files)
        self._pf = [pq.ParquetFile(f) for f in self.files]   # footers only
        self._file_cum = [0]
        self._rg_cum = []
        for pf in self._pf:
            self._file_cum.append(self._file_cum[-1] + pf.metadata.num_rows)
            offs = [0]
            for i in range(pf.num_row_groups):
                offs.append(offs[-1] + pf.metadata.row_group(i).num_rows)
            self._rg_cum.append(offs)

    def __len__(self):
        return self._file_cum[-1]

    def tasks(self):
        out = []
        for pf in self._pf:
            out.extend(pf.read(columns=["task"]).column("task").to_pylist())
        return out

    def __getitem__(self, idx):
        fi = bisect.bisect_right(self._file_cum, idx) - 1
        local = idx - self._file_cum[fi]
        offs = self._rg_cum[fi]
        rg = bisect.bisect_right(offs, local) - 1
        tbl = self._pf[fi].read_row_group(rg, columns=self.COLUMNS)
        r = local - offs[rg]
        return {c: tbl.column(c)[r].as_py() for c in self.COLUMNS}


class OmniEditDataset(Dataset):
    """Task-balanced sampler; `idx` is ignored (every __getitem__ draws a fresh sample)."""

    TASKS = ("swap", "attribute_modification", "addition", "removal", "env", "style")
    NUM_TURNS = (1, 2, 3, 4)   # used only when the rollout curriculum is off

    def __init__(self, root, split="train", seed=0, prompt_index=1):
        files = sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no {root}/data/{split}-*.parquet")
        self.rows = ParquetRows(files)
        self.prompt_index = prompt_index      # edited_prompt_list[1] = the detailed prompt
        self.rng = random.Random(seed)
        self.task_to_indices = {}
        for i, t in enumerate(self.rows.tasks()):
            self.task_to_indices.setdefault(t, []).append(i)
        self.tasks = ([t for t in self.TASKS if t in self.task_to_indices]
                      + [t for t in self.task_to_indices if t not in self.TASKS])

    def __len__(self):
        return len(self.rows)

    def _instruction(self, prompts):
        if not prompts:
            return ""
        i = self.prompt_index if 0 <= self.prompt_index < len(prompts) else len(prompts) - 1
        return prompts[i]

    def __getitem__(self, idx):
        task = self.rng.choice(self.tasks)
        row = self.rng.choice(self.task_to_indices[task])
        num_turns = self.rng.choices(self.NUM_TURNS, weights=[1] * len(self.NUM_TURNS), k=1)[0]
        ex = self.rows[row]
        cell = ex["src_img"]
        src = (Image.open(io.BytesIO(cell["bytes"])) if cell.get("bytes") is not None
               else Image.open(cell["path"]))
        return {
            "source_image": src.convert("RGB"),
            "instruction": self._instruction(ex["edited_prompt_list"]),
            "num_turns": num_turns,
            "task": task,
            "omni_edit_id": ex["omni_edit_id"],
        }


def _collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0]}


def build_dataloader(args, rank=0):
    # per-rank seed -> each DDP process draws a different sample stream. num_workers=0:
    # the GPU rollout dominates, and forked workers would duplicate the sampler RNG.
    ds = OmniEditDataset(args.data_root, args.split, seed=args.data_seed + rank)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate)
