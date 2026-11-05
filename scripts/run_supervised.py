"""Interruption-tolerant training launcher (supervisor/resubmission loop).

WHY (2026-07-17): the Week-3 pilot batch was killed by the task harness
~1h50m in — an infrastructure lifetime cap, not a training failure. The
Weeks 5-7 grid needs 50-100 unattended runs, so every launch must be
capped SAFELY UNDER the observed kill threshold and automatically resumed
from the last checkpoint until the target step count is reached.

Takes the same training flags as train_baseline.py and passes them
through verbatim; supervisor-specific flags:
  --launch-timeout   seconds per launch (default 3600 — well under the
                     ~1h50m observed harness kill)
  --max-stalls       consecutive launches with NO checkpoint progress
                     before aborting (default 3; guards crash loops)

Each relaunch passes --resume, which falls back to a fresh start when no
checkpoint exists yet, so the first launch needs no special casing.

  python scripts/run_supervised.py --skill arith --skill-exposure-frac \\
      0.20 --delay-onset 0.50 --launch-timeout 3600
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import CHECKPOINT_DIR, TrainConfig  # noqa: E402

TRAIN_SCRIPT = Path(__file__).parent / "train_baseline.py"


def latest_step(ckpt_dir: Path) -> int:
    hits = sorted(ckpt_dir.glob("step_*.pt"))
    return int(hits[-1].stem.split("_")[1]) if hits else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-timeout", type=int, default=3600)
    ap.add_argument("--max-stalls", type=int, default=3)
    # mirror the train_baseline flags that determine run_id / total steps
    # (defaults must match TrainConfig — both files own this contract)
    defaults = TrainConfig()
    ap.add_argument("--skill", default="none")
    ap.add_argument("--delay-onset", type=float, default=None)
    ap.add_argument("--skill-exposure-frac", type=float,
                    default=defaults.skill_exposure_frac)
    ap.add_argument("--total-steps", type=int, default=defaults.total_steps)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--run-name", default=defaults.run_name)
    ap.add_argument("--model-size", default=defaults.model_size_tag,
                    help="architecture size tag (config.py registry); must "
                         "match what train_baseline trains so the supervisor "
                         "watches the right run_id")
    ap.add_argument("--run-suffix", default="")
    # run_id-determining: the supervisor must reconstruct the SAME run_id the
    # trainer will use, or it watches the wrong checkpoint dir
    ap.add_argument("--lr-schedule", default=defaults.lr_schedule)
    args, passthrough = ap.parse_known_args()

    delay = args.delay_onset
    if args.skill != "none" and delay is None:
        delay = 0.0
    cfg = TrainConfig(
        skill=args.skill, delay_onset=delay,
        skill_exposure_frac=args.skill_exposure_frac,
        total_steps=args.total_steps, seed=args.seed,
        run_name=args.run_name, run_suffix=args.run_suffix,
        model_size_tag=args.model_size,
        lr_schedule=args.lr_schedule,
    )
    ckpt_dir = CHECKPOINT_DIR / cfg.run_id

    train_args = [
        "--skill", args.skill,
        "--skill-exposure-frac", str(args.skill_exposure_frac),
        "--total-steps", str(args.total_steps),
        "--seed", str(args.seed),
        "--run-name", args.run_name,
        "--model-size", args.model_size,
        "--lr-schedule", args.lr_schedule,
        *(["--delay-onset", str(delay)] if delay is not None else []),
        *(["--run-suffix", args.run_suffix] if args.run_suffix else []),
        *passthrough,
    ]

    stalls = 0
    launch = 0
    # INVARIANT: high_water is scoped to THIS run_id (one supervisor
    # process = one run_id; ckpt_dir above is fixed for the process
    # lifetime). If a future grid driver ever loops multiple run_ids
    # inside one process, high_water/stalls/launch MUST reset per run_id —
    # otherwise a fresh run starting at step 0 after another run reached
    # step N would falsely trip the regression abort. Prefer one
    # subprocess invocation of this script per run_id instead.
    high_water = 0
    while True:
        before = latest_step(ckpt_dir)
        if before < high_water:
            sys.exit(f"[supervisor] checkpoint REGRESSION: latest step "
                     f"{before} < previously observed {high_water} — files "
                     "deleted or path changed; aborting rather than "
                     "silently restarting under the same run_id")
        high_water = before
        if before >= args.total_steps:
            break
        launch += 1
        print(f"[supervisor] launch {launch}: {cfg.run_id} at step {before}"
              f"/{args.total_steps} (timeout {args.launch_timeout}s)",
              flush=True)
        # fresh start is only legitimate while NO checkpoint has ever been
        # observed; once one exists, retries must fail loudly on a missing
        # checkpoint instead of quietly restarting from random init
        resume_flag = ["--resume-strict"] if high_water > 0 else []
        cmd = [sys.executable, str(TRAIN_SCRIPT), *train_args, *resume_flag]
        t0 = time.time()
        timed_out = False
        try:
            proc = subprocess.run(cmd, timeout=args.launch_timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            code = None
        after = latest_step(ckpt_dir)
        print(f"[supervisor] launch {launch} ended after "
              f"{(time.time()-t0)/60:.1f} min "
              f"({'timeout' if timed_out else f'exit {code}'}), "
              f"step {before} -> {after}", flush=True)
        if after >= args.total_steps:
            break
        if not timed_out and code == 0 and after < args.total_steps:
            sys.exit(f"[supervisor] training exited 0 at step {after} < "
                     f"{args.total_steps} — config mismatch? aborting")
        if after <= before:
            stalls += 1
            if stalls >= args.max_stalls:
                sys.exit(f"[supervisor] {stalls} launches with no progress "
                         f"(stuck at step {after}) — aborting")
        else:
            stalls = 0
    print(f"[supervisor] {cfg.run_id} complete at step "
          f"{latest_step(ckpt_dir)}")


if __name__ == "__main__":
    main()
