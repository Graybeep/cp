"""Post-hoc evaluation of saved checkpoints — full eval sets, all BLiMP
paradigms, optional per-word vocab breakdown.

This is the Week 3 gate workhorse: run with --per-word on pilot runs to
judge whether the vocab control is cleanly flat or just underpowered
(see CLAUDE.md Week 3 EXIT GATE).

Run:  python scripts/eval_checkpoints.py <run_id> [--per-word] [--n 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.checkpoint import load_checkpoint  # noqa: E402
from reopen.config import CHECKPOINT_DIR, LOG_DIR, ModelConfig  # noqa: E402
from reopen.evals import EvalHarness  # noqa: E402
from reopen.incremental import IncrementalCSV, IncrementalJSONL, archive  # noqa: E402
from reopen.model import GPT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--n", type=int, default=0,
                    help="instances per skill (0 = full eval sets)")
    ap.add_argument("--per-word", action="store_true",
                    help="write per-word vocab breakdown per checkpoint")
    ap.add_argument("--primaries-only", action="store_true",
                    help="skip the five BLiMP contrast paradigms")
    ap.add_argument("--fresh", action="store_true",
                    help="discard existing posthoc outputs and re-eval all "
                         "checkpoints (REQUIRED after the eval sets change; "
                         "prior outputs are archived as .bak, not deleted)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt_dir = CHECKPOINT_DIR / args.run_id
    paths = sorted(ckpt_dir.glob("step_*.pt"))
    assert paths, f"no checkpoints under {ckpt_dir}"

    harness = EvalHarness(
        device=args.device,
        n_per_skill=args.n or None,
        full_blimp=not args.primaries_only,
    )

    out_dir = LOG_DIR / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    per_word_file = (out_dir / "posthoc_vocab_per_word.jsonl") \
        if args.per_word else None
    out_csv = out_dir / "posthoc_evals.csv"

    # Incremental + resumable via reopen.incremental (a background kill
    # mid-sweep loses at most one checkpoint; reruns skip completed work;
    # writes are atomic). --fresh archives instead of deleting: an
    # eval-set change makes old rows stale, but destroying them silently
    # cost this project its original per-word record on 2026-07-17.
    if args.fresh:
        for f in (per_word_file, out_csv):
            if f is not None:
                archive(f)
    csv_out = IncrementalCSV(out_csv, key="step")
    pw_out = IncrementalJSONL(per_word_file, key="step") \
        if per_word_file is not None else None
    done_steps = csv_out.done
    if args.per_word:
        # only skip checkpoints complete in BOTH outputs
        done_steps = done_steps & pw_out.done
    if done_steps:
        print(f"resuming: {len(done_steps)} checkpoints already evaluated")

    model = None
    for p in paths:
        if int(p.stem.split("_")[1]) in done_steps:
            continue
        ckpt = load_checkpoint(p, device="cpu")
        if model is None:
            mcfg = ModelConfig(**ckpt["model_config"])
            model = GPT(mcfg).to(args.device)
        model.load_state_dict(ckpt["model"])
        res = harness.evaluate(model, per_word=args.per_word)
        row = {
            "run_id": args.run_id,
            "step": ckpt["step"],
            "arith_accuracy": res["arith_accuracy"],
            "syntax_accuracy": res["syntax_accuracy"],
            "syntax_accuracy_local": res["syntax_accuracy_local"],
            "vocab_control_accuracy": res["vocab_control_accuracy"],
            "vocab_control_logprob": res["vocab_control_logprob"],
            **{f"blimp_{k}": v for k, v in res["blimp"].items()},
        }
        print(", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items() if k != "run_id"))
        # per-word first, CSV second: the resume rule skips only steps
        # present in BOTH, so a kill between the two writes re-does the
        # checkpoint rather than losing its per-word half
        if pw_out is not None and "vocab_per_word" in res:
            pw_out.append({"step": ckpt["step"],
                           "per_word": res["vocab_per_word"]})
        csv_out.append(row)

    print(f"wrote {out_csv} ({len(csv_out.rows)} checkpoints)")


if __name__ == "__main__":
    main()
