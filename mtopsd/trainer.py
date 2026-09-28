"""MT-OPSD training loop.

Every step:
  1. draw a clean source I_0 and an instruction e (OmniEdit);
  2. roll the CURRENT student out with the identity instruction e_id for num_turns-1 turns
     (no grad, same sampler as inference) -> rollout state I_prev, whose deviation from
     I_0 is error the model introduced itself;
  3. with probability edit_ratio take the editing branch (sparse on-policy velocity matching
     against the frozen teacher conditioned on I_0), otherwise the identity branch (I_prev
     is its own target under e_id).
The rollout curriculum raises num_turns once rollouts stay clean, and with --gate_promote
the teacher is hot-swapped to checkpoints promoted by the gate worker.

Step counters (max_steps, save_every, eval_every, log_every, checkpoint names) are GLOBAL
samples: one optimizer step consumes num_gpus samples.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import signal
import time

import numpy as np
import torch
import yaml

from .drift_eval import drift_eval
from .mt_eval import load_sentinels, plain_vs_emu


# --------------------------------------------------------------------------- checkpoints
def checkpoint_dirs(output_dir):
    """Complete checkpoint-N dirs (training_state.pt written), newest first."""
    found = []
    for d in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", d)
        if m and os.path.isfile(os.path.join(d, "training_state.pt")):
            found.append((int(m.group(1)), d))
    return [d for _, d in sorted(found, reverse=True)]


def save_checkpoint(model, optimizer, step, args, ckpt_dir, wandb_run_id, edit_rng, curriculum):
    from peft.utils import get_peft_model_state_dict

    model.save_lora(ckpt_dir)
    with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
        json.dump({"model": args.model, "rank": args.rank, "lora_alpha": args.lora_alpha,
                   "target_modules": list(args.target_modules), "train_area": args.train_area,
                   "steps": args.rollout_steps, "cfg": args.rollout_cfg}, f, indent=1)
    state = {
        "step": step,
        "lora": get_peft_model_state_dict(model.transformer),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "py_rng": random.getstate(),
        "edit_rng": edit_rng,
        "curriculum": curriculum,
        "wandb_run_id": wandb_run_id,
        "args": {k: v for k, v in vars(args).items() if k not in ("device", "proc")},
        "gate_teacher_step": model.gate_teacher_step,
    }
    # atomic: a preemption mid-write must not leave a corrupt file for --resume
    path = os.path.join(ckpt_dir, "training_state.pt")
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[ckpt] {ckpt_dir}", flush=True)


def maybe_resume(args, model, optimizer):
    """Load the newest readable checkpoint (LoRA, optimizer, RNGs, promoted teacher).
    Returns (opt_step, wandb_run_id, edit_rng_state, curriculum_state)."""
    if args.resume in (None, "", "none"):
        return 0, None, None, None
    cands = [args.resume] if args.resume != "auto" else checkpoint_dirs(args.output_dir)
    state = ckpt = None
    for c in cands:
        try:
            state = torch.load(os.path.join(c, "training_state.pt"), map_location="cpu",
                               weights_only=False)
            ckpt = c
            break
        except Exception as e:  # noqa: BLE001  (interrupted save -> try an older one)
            print(f"[resume] {c} unreadable ({type(e).__name__}); trying older", flush=True)
    if state is None:
        print("[resume] no checkpoint -> fresh start", flush=True)
        return 0, None, None, None
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model.transformer, state["lora"])
    optimizer.load_state_dict(state["optimizer"])
    for st in optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(args.device)
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["py_rng"])
    gts = state.get("gate_teacher_step", -1)
    if gts is not None and gts >= 0:
        model.promote_teacher(os.path.join(args.output_dir, f"checkpoint-{gts}"),
                              args.rank, args.lora_alpha, args.target_modules)
        model.gate_teacher_step = gts
    print(f"[resume] {ckpt} @ opt-step {state['step']}", flush=True)
    return int(state["step"]), state.get("wandb_run_id"), state.get("edit_rng"), state.get("curriculum")


def poll_gate(model, args, accelerator, log):
    """Rank 0 looks for the newest promote=true verdict newer than the current teacher and
    broadcasts it, so every rank swaps the same checkpoint at the same step boundary."""
    target = -1
    if accelerator.is_main_process:
        for f in glob.glob(os.path.join(args.output_dir, "gate", "verdict-*.json")):
            try:
                v = json.load(open(f))
            except Exception:  # noqa: BLE001  (torn write; retry next poll)
                continue
            s = int(v.get("step", -1))
            if (v.get("promote") and s > model.gate_teacher_step and s > target
                    and os.path.isfile(os.path.join(args.output_dir, f"checkpoint-{s}",
                                                    "training_state.pt"))):
                target = s
    if accelerator.num_processes > 1:
        t = torch.tensor([target], dtype=torch.long, device=accelerator.device)
        torch.distributed.broadcast(t, src=0)
        target = int(t.item())
    if target >= 0:
        model.promote_teacher(os.path.join(args.output_dir, f"checkpoint-{target}"),
                              args.rank, args.lora_alpha, args.target_modules)
        model.gate_teacher_step = target
        log(f"[gate] teacher <- checkpoint-{target}")


# --------------------------------------------------------------------------- loop
def run_training(args, model, loader, optimizer, trainable, accelerator, eval_img=None,
                 start_step=0, wandb_run_id=None, edit_rng_state=None, curriculum_state=None):
    is_main = accelerator.is_main_process
    world = accelerator.num_processes
    os.makedirs(args.output_dir, exist_ok=True)
    if is_main:
        with open(os.path.join(args.output_dir, "config_resolved.yaml"), "w") as f:
            yaml.safe_dump({k: v for k, v in vars(args).items()
                            if k not in ("device", "proc")},
                           f, default_flow_style=False, sort_keys=False)
    log_path = os.path.join(args.output_dir, "train_log.txt")

    def log(msg):
        if is_main:
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

    run = None
    if args.wandb and is_main:
        import wandb
        kw = dict(project=args.wandb_project, entity=args.wandb_entity, config=vars(args),
                  name=args.wandb_run_name or os.path.basename(args.output_dir.rstrip("/")))
        if wandb_run_id:
            kw.update(id=wandb_run_id, resume="allow")
        run = wandb.init(**kw)

    def to_opt(n):
        return max(1, n // world)

    max_opt, save_opt, log_opt = to_opt(args.max_steps), to_opt(args.save_every), to_opt(args.log_every)
    eval_opt = to_opt(args.eval_every) if args.eval_every else 0
    sentinels = load_sentinels(args.mt_sentinels) if args.mt_eval and is_main else []

    def maybe_eval(gstep):
        if eval_img is not None and args.eval_every and is_main:
            model.transformer.eval()
            out = os.path.join(args.output_dir, "eval", f"drift_step{gstep}.png")
            drifts = drift_eval(model, eval_img, out, rounds=args.drift_rounds,
                                steps=args.drift_steps, prompt=args.noop_prompt,
                                cfg=args.drift_cfg, seed=args.eval_seed)
            model.transformer.train()
            log(f"[eval step {gstep}] identity drift per round = {drifts} -> {out}")
            if run is not None:
                import wandb
                d = {f"eval/drift_r{i + 1}": x for i, x in enumerate(drifts)}
                d["eval/drift_strip"] = wandb.Image(out)
                run.log(d, step=gstep)
        if sentinels and args.eval_every and is_main:
            model.transformer.eval()
            for sid, src, turns in sentinels:
                out = os.path.join(args.output_dir, "eval", f"mt_step{gstep}_{sid}")
                drifts, cfs = plain_vs_emu(model, src, turns, out, steps=args.drift_steps,
                                           cfg=args.drift_cfg, seed=args.eval_seed)
                log(f"[eval step {gstep}] mt[{sid}] plain-vs-emu drift per turn = {drifts}")
                log(f"[eval step {gstep}] mt[{sid}] colorfulness per turn = {cfs}")
                if run is not None:
                    import wandb
                    d = {f"eval/mt_drift_t{i + 1}_{sid}": x for i, x in enumerate(drifts)}
                    d.update({f"eval/mt_cf_t{i + 1}_{sid}": x for i, x in enumerate(cfs)})
                    d[f"eval/mt_strip_plain_{sid}"] = wandb.Image(os.path.join(out, "strip_plain.png"))
                    d[f"eval/mt_strip_emu_{sid}"] = wandb.Image(os.path.join(out, "strip_emu.png"))
                    run.log(d, step=gstep)
            model.transformer.train()
        accelerator.wait_for_everyone()

    # rollout curriculum state (per rank; each rank gates on its own rollouts)
    cap, consec = args.turns_curriculum_init, 0
    if args.turns_curriculum and curriculum_state is not None:
        cap, consec = curriculum_state["cap"], curriculum_state["consec"]

    # branch choice: SAME seed on every rank -> all ranks take the same branch each step,
    # so the DDP graph (used parameters) matches across ranks
    rng_edit = random.Random(args.seed + 100003)
    if edit_rng_state is not None:
        rng_edit.setstate(edit_rng_state)
    rng_query = random.Random(args.seed + 424243 + args.proc)   # OPD query positions (per rank)

    preempt = {"flag": False}

    def on_term(signum, frame):
        preempt["flag"] = True
        log(f"[signal] {signum}: checkpoint at the next step boundary")

    signal.signal(signal.SIGTERM, on_term)

    def do_ckpt(opt_step):
        if is_main:
            save_checkpoint(model, optimizer, opt_step, args,
                            os.path.join(args.output_dir, f"checkpoint-{opt_step * world}"),
                            run.id if run else None, rng_edit.getstate(),
                            {"cap": cap, "consec": consec})
        accelerator.wait_for_everyone()

    log(f"[run] {'resume' if start_step else 'start'} @ {start_step * world}/{args.max_steps} samples, "
        f"{world} gpu(s), model={args.model} lr={args.lr} edit_ratio={args.edit_ratio} "
        f"curriculum={'%d->%d' % (cap, args.turns_curriculum_max) if args.turns_curriculum else 'off'} "
        f"gate_promote={args.gate_promote}")

    model.transformer.train()
    if start_step == 0:
        maybe_eval(0)

    step = start_step
    run_all = run_edit = run_noop = 0.0
    n_edit = n_noop = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    while step < max_opt:
        if preempt["flag"]:
            do_ckpt(step)
            log(f"[signal] checkpointed @ {step * world} samples; exiting for requeue")
            if run is not None:
                run.finish()
            return
        batch = next(data_iter)
        src, instr = batch["source_image"][0], batch["instruction"][0]
        num_turns = cap if args.turns_curriculum else min(int(batch["num_turns"][0]),
                                                          args.max_num_turns)

        # 1) self-rollout with the identity instruction
        i_prev = model.rollout(src, num_turns - 1, steps=args.rollout_steps,
                               cfg=args.rollout_cfg, noop_prompt=args.noop_prompt)

        # 2) curriculum: deepen once `patience` consecutive rollouts drift <= threshold
        if args.turns_curriculum and cap < args.turns_curriculum_max:
            ref = src.convert("RGB")
            ref = ref if ref.size == i_prev.size else ref.resize(i_prev.size)
            drift = float(np.abs(np.asarray(i_prev.convert("RGB"), np.float32)
                                 - np.asarray(ref, np.float32)).mean())
            consec = consec + 1 if drift <= args.turns_curriculum_drift else 0
            if consec >= args.turns_curriculum_patience:
                cap, consec = cap + 1, 0
                log(f"[curriculum] num_turns -> {cap} @ ~{step * world} samples")

        # 3) editing branch (OPD) or identity branch
        do_edit = args.edit_ratio > 0 and rng_edit.random() < args.edit_ratio
        if do_edit:
            # backward + cross-rank all-reduce happen inside; returns a float
            lv = model.compute_opd_loss(instr, src, i_prev, num_steps=args.opd_steps,
                                        cfg=args.opd_cfg, k=args.opd_k,
                                        query_bias=args.opd_query_bias, query_rng=rng_query)
            run_edit += lv
            n_edit += 1
        else:
            loss = model.compute_noop_loss(i_prev, args)
            accelerator.backward(loss * args.noop_loss_scale)
            lv = loss.item()
            run_noop += lv
            n_noop += 1
        run_all += lv

        if args.max_grad_norm > 0:
            if do_edit:   # grads were all-reduced manually, not by the DDP reducer
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            else:
                accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        gstep = step * world

        if step % log_opt == 0:
            n = n_edit + n_noop
            dt = (time.time() - t0) / log_opt
            le, ln = run_edit / max(n_edit, 1), run_noop / max(n_noop, 1)
            log(f"[step {gstep:>6}/{args.max_steps}] loss={run_all / max(n, 1):.5f} "
                f"edit_frac={n_edit / max(n, 1):.2f} loss_edit={le:.5f} loss_noop={ln:.5f} "
                f"{dt:.1f}s/step num_turns={num_turns} teacher={model.gate_teacher_step}")
            if run is not None:
                run.log({"train/loss": run_all / max(n, 1), "train/loss_edit": le,
                         "train/loss_noop": ln, "train/num_turns": num_turns,
                         "train/sec_per_step": dt,
                         "train/teacher_step": model.gate_teacher_step}, step=gstep)
            run_all = run_edit = run_noop = 0.0
            n_edit = n_noop = 0
            t0 = time.time()

        if eval_opt and step % eval_opt == 0:
            maybe_eval(gstep)
        if step % save_opt == 0:
            do_ckpt(step)
        if args.gate_promote and step % args.gate_poll_every == 0:
            poll_gate(model, args, accelerator, log)

    if is_main:
        model.save_lora(os.path.join(args.output_dir, "final"))
    accelerator.wait_for_everyone()
    maybe_eval(step * world)
    log("[run] done")
    if run is not None:
        run.finish()
