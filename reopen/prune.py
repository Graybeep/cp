"""Checkpoint retention — reclaim the intermediate-checkpoint storage leak.

WHY (2026-07-21): every full run saves all 50 checkpoints (step 100..5000,
interval 100) at ~278MB each => ~13GB/run. Across ~117 runs that reached
379GB. But the research design only ever *reads* two checkpoints per run:

  * step 2500  — the Week-4 common-checkpoint / fork point
  * step 5000  — the final model (Detect final-accuracy, eval)

Every intermediate checkpoint's information (per-step skill accuracy + val
perplexity) is already persisted in logs/<run_id>/metrics.csv and
eval_details.jsonl, so deleting the intermediate *weights* loses no research
data. Weeks-5-7 grid would need 600GB+ at the old policy; keeping only
{2500, 5000} it is ~30GB.

SAFETY MODEL (this tool never strands a resumable or live run):
  * KEEP_STEPS default {2500, 5000}. Keep whichever of them exist.
  * If a run has NO final (step 5000) it is INCOMPLETE — additionally keep its
    highest-step checkpoint so run_supervised can still resume it.
  * Never delete a run's only checkpoint.
  * Skip any run whose newest checkpoint was modified within --active-window
    minutes (a run currently training), plus any --exclude'd run_id.
  * Dry-run by default; deletion requires --apply.

Reusable as a library (`prune_run`) so the Weeks-5-7 driver can call it on
run completion instead of letting storage refill.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import CHECKPOINT_DIR  # noqa: E402

# Retain every delay-onset fraction (25/50/75%) plus the final. Reopen forks
# from a baseline at whatever onset showed a gap, so a completed baseline must
# keep ALL plausible fork/common-checkpoint points — never auto-delete a
# future fork source. ~4 ckpts (~1.1GB) vs 50 (~13GB): still ~90% reclaim.
ONSET_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_KEEP_STEPS = (2500, 5000)  # standalone-CLI convenience for 5k runs


def onset_keep_steps(total_steps: int) -> tuple[int, ...]:
    """Fork-safe keep-set for a completed run of `total_steps` steps."""
    return tuple(sorted({int(round(f * total_steps)) for f in ONSET_FRACTIONS}))


def _step_of(p: Path) -> int:
    return int(p.stem.split("_")[1])


def plan_run(run_dir: Path, keep_steps=DEFAULT_KEEP_STEPS) -> tuple[list[Path], list[Path]]:
    """Return (keep, delete) checkpoint lists for one run directory.

    Pure/no-IO-side-effect: caller decides whether to act. Encodes the full
    safety policy so both the CLI and an in-process caller share it.

    Two safety invariants:
      * INCOMPLETE runs (final step absent) are never pruned — every
        checkpoint is kept so the run stays fully resumable and never
        restarts from the beginning.
      * Each keep target SNAPS to the nearest existing checkpoint. Onset
        fractions (25/75% = 1250/3750) don't land on the 100-step grid, so an
        exact-match keep would silently drop the checkpoint closest to a fork
        point. Snapping guarantees a fork source near every onset survives.
    """
    ckpts = sorted(run_dir.glob("step_*.pt"), key=_step_of)
    if not ckpts:
        return [], []
    existing = {_step_of(p): p for p in ckpts}
    final = max(keep_steps)
    if final not in existing:
        return ckpts, []  # incomplete -> keep everything (resume safety)
    keep: set[Path] = set()
    for target in keep_steps:
        if target in existing:
            keep.add(existing[target])
        else:  # nearest existing checkpoint; tie -> the later step
            nearest = min(existing, key=lambda s: (abs(s - target), -s))
            keep.add(existing[nearest])
    delete = [p for p in ckpts if p not in keep]
    return sorted(keep, key=_step_of), delete


def prune_run(run_dir: Path, keep_steps=DEFAULT_KEEP_STEPS,
              apply: bool = False) -> tuple[int, int]:
    """Prune one run. Returns (files_deleted, bytes_deleted). apply=False =
    dry run (counts only). Intended for both the CLI and on-completion calls
    from the training drivers."""
    _keep, delete = plan_run(run_dir, keep_steps)
    n = sum(1 for _ in delete)
    b = sum(p.stat().st_size for p in delete)
    if apply:
        for p in delete:
            p.unlink()
    return n, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-steps", default="2500,5000",
                    help="comma-separated steps to always keep (default 2500,5000)")
    ap.add_argument("--active-window", type=int, default=15,
                    help="skip runs whose newest ckpt was touched within N "
                         "minutes (guards a live run; default 15)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="run_id to skip (repeatable)")
    ap.add_argument("--run", action="append", default=[],
                    help="limit to these run_id(s) (default: all runs)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry run)")
    args = ap.parse_args()

    keep_steps = tuple(int(s) for s in args.keep_steps.split(","))
    now = time.time()
    run_dirs = ([CHECKPOINT_DIR / r for r in args.run] if args.run
                else sorted(d for d in CHECKPOINT_DIR.iterdir() if d.is_dir()))

    tot_n = tot_b = 0
    skipped = []
    for d in run_dirs:
        rid = d.name
        if rid in args.exclude:
            skipped.append((rid, "excluded")); continue
        ckpts = list(d.glob("step_*.pt"))
        if not ckpts:
            continue
        newest_age = (now - max(p.stat().st_mtime for p in ckpts)) / 60.0
        if newest_age < args.active_window:
            skipped.append((rid, f"active ({newest_age:.1f}m old)")); continue
        n, b = prune_run(d, keep_steps, apply=args.apply)
        if n:
            print(f"  {'DEL' if args.apply else 'would del'} {n:>2} ckpts "
                  f"({b/1024**3:5.1f} GB)  {rid}")
        tot_n += n; tot_b += b

    print("-" * 60)
    verb = "DELETED" if args.apply else "WOULD DELETE"
    print(f"{verb}: {tot_n} checkpoints, {tot_b/1024**3:.1f} GB "
          f"(keep steps {keep_steps})")
    for rid, why in skipped:
        print(f"  skipped: {rid} — {why}")
    if not args.apply:
        print("\n(dry run — re-run with --apply to delete)")


if __name__ == "__main__":
    main()
