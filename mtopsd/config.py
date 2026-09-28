"""Command-line / YAML configuration.

Precedence: CLI flag > YAML (`--config`) > argparse default. YAML keys must match
the argparse destination names below.
"""
from __future__ import annotations

import argparse

import yaml


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="YAML config; CLI flags override it.")

    g = parser.add_argument_group("model")
    g.add_argument("--model", required=False, default=None,
                   help="backbone name, see mtopsd.models.MODELS.")
    g.add_argument("--pretrained_path", default=None,
                   help="local path or HF repo id of the base model (default: the backbone's HF id).")
    g.add_argument("--train_area", type=int, default=512 * 512,
                   help="working pixel area (aspect preserved) for condition, output, rollout and "
                        "eval. Inference must use the same area.")

    g = parser.add_argument_group("data")
    g.add_argument("--data_root", default=None,
                   help="OmniEdit-Filtered-1.2M root holding data/<split>-*.parquet.")
    g.add_argument("--split", default="train")
    g.add_argument("--data_seed", type=int, default=0)
    g.add_argument("--max_num_turns", type=int, default=4,
                   help="rollout depth cap when --turns_curriculum is off (num_turns ~ U{1..4}).")

    g = parser.add_argument_group("lora")
    g.add_argument("--rank", type=int, default=32)
    g.add_argument("--lora_alpha", type=int, default=64)
    g.add_argument("--target_modules", nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])

    g = parser.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=3e-4, help="constant learning rate.")
    g.add_argument("--weight_decay", type=float, default=0.0)
    g.add_argument("--adam_beta1", type=float, default=0.9)
    g.add_argument("--adam_beta2", type=float, default=0.999)
    g.add_argument("--adam_eps", type=float, default=1e-8)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--max_steps", type=int, default=2000,
                   help="GLOBAL sample budget. Optimizer steps = max_steps / num_gpus; "
                        "save_every / eval_every / log_every use the same unit.")
    g.add_argument("--logit_mean", type=float, default=0.0,
                   help="logit-normal timestep sampling for the identity (flow-matching) loss.")
    g.add_argument("--logit_std", type=float, default=1.0)

    g = parser.add_argument_group("self-rollout")
    g.add_argument("--rollout_steps", type=int, default=30,
                   help="sampling steps of the identity rollout (= deployment steps).")
    g.add_argument("--rollout_cfg", type=float, default=4.0, help="true CFG scale of the rollout.")
    g.add_argument("--noop_prompt", default="make everything unchange",
                   help="the identity instruction e_id.")

    g = parser.add_argument_group("loss branches")
    g.add_argument("--edit_ratio", type=float, default=0.667,
                   help="P(editing branch) per step; the rest are identity-branch steps (2:1).")
    g.add_argument("--noop_loss_scale", type=float, default=10.0,
                   help="multiplier on the identity loss before backward (logged value stays raw).")

    g = parser.add_argument_group("on-policy self-distillation (editing branch)")
    g.add_argument("--opd_steps", type=int, default=10, help="ODE steps of the student rollout.")
    g.add_argument("--opd_cfg", type=float, default=4.0, help="true CFG of student and teacher.")
    g.add_argument("--opd_k", type=int, default=2, help="teacher queries per rollout.")
    g.add_argument("--opd_query_bias", default="mid_t",
                   choices=["low_t", "mid_t", "high_t", "uniform"],
                   help="where the queries fall along the rollout: low_t = Beta(5,2) (near clean), "
                        "mid_t = Beta(5,5), high_t = Beta(2,5) (near noise), uniform.")

    g = parser.add_argument_group("rollout curriculum")
    g.add_argument("--turns_curriculum", action="store_true", default=False,
                   help="grow the rollout depth adaptively: num_turns = cap every step, and cap += 1 "
                        "once the per-step rollout drift (mean |I_prev - I_0|, 0-255) stays <= "
                        "--turns_curriculum_drift for --turns_curriculum_patience steps in a row.")
    g.add_argument("--turns_curriculum_init", type=int, default=2)
    g.add_argument("--turns_curriculum_max", type=int, default=10)
    g.add_argument("--turns_curriculum_drift", type=float, default=10.0)
    g.add_argument("--turns_curriculum_patience", type=int, default=10)

    g = parser.add_argument_group("gated teacher promotion")
    g.add_argument("--gate_promote", action="store_true", default=False,
                   help="hot-swap the frozen teacher to checkpoints the gate worker (gate_worker.py, "
                        "a separate job) promotes via <output_dir>/gate/verdict-*.json.")
    g.add_argument("--gate_poll_every", type=int, default=10,
                   help="poll the verdict dir every N optimizer steps.")
    g.add_argument("--gate_sr_bar", type=int, default=2,
                   help="[read by gate_worker.py] success-led promotion: the candidate must pass "
                        ">= this many MORE gate sessions than the teacher.")
    g.add_argument("--gate_col_unlock", type=int, default=5,
                   help="[read by gate_worker.py] stability-led promotion: >= this many FEWER "
                        "collapsed sessions with success flat (|dSR| <= 1).")
    g.add_argument("--gate_min_step", type=int, default=0,
                   help="[read by gate_worker.py] checkpoints before this many GLOBAL samples "
                        "are never judged (their gate scores are too noisy to promote on).")

    g = parser.add_argument_group("in-training identity drift eval")
    g.add_argument("--eval_img", default="assets/drift_eval_input.png", help="'' disables it.")
    g.add_argument("--eval_every", type=int, default=50, help="GLOBAL samples; 0 disables it.")
    g.add_argument("--drift_rounds", type=int, default=10)
    g.add_argument("--drift_steps", type=int, default=30)
    g.add_argument("--drift_cfg", type=float, default=4.0)
    g.add_argument("--eval_seed", type=int, default=42)
    g.add_argument("--mt_eval", action="store_true", default=False,
                   help="at every eval point also run the sentinel sessions with and without "
                        "Emu thresholding and log their per-turn difference and colorfulness "
                        "(mtopsd/mt_eval.py). Uses drift_steps / drift_cfg / eval_seed.")
    g.add_argument("--mt_sentinels", default="assets/mt_sentinels/sentinels.json")

    g = parser.add_argument_group("run")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output_dir", default=None)
    g.add_argument("--log_every", type=int, default=10, help="GLOBAL samples.")
    g.add_argument("--save_every", type=int, default=50, help="GLOBAL samples.")
    g.add_argument("--resume", default="auto",
                   help="'auto' = latest checkpoint-* in output_dir if any; 'none'; or a checkpoint dir.")
    g.add_argument("--wandb", action="store_true", default=False)
    g.add_argument("--wandb_project", default="mt-opsd")
    g.add_argument("--wandb_run_name", default=None)
    g.add_argument("--wandb_entity", default=None)
    return parser


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_with_config(parser: argparse.ArgumentParser, argv):
    """Two-phase parse: YAML values become parser defaults, explicit CLI flags still win."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        cfg = load_yaml(known.config)
        unknown = sorted(set(cfg) - {a.dest for a in parser._actions})
        if unknown:
            parser.error(f"unknown keys in {known.config}: {unknown}")
        parser.set_defaults(**cfg)
    return parser.parse_args(argv)
