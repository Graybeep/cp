"""Week 4 fork launcher: resume a source run's checkpoint into a NEW
run_id with intervention triggers attached.

COMMON-CHECKPOINT RULE (locked, CLAUDE.md Week 4): every intervention run
for a given skill/delay combo resumes from the IDENTICAL checkpoint file —
same weights, same optimizer state, same step — and shares its post-resume
data stream except where the intervention itself changes it (cram). This
script enforces that by construction:

- the TrainConfig is reconstructed FROM THE SOURCE CHECKPOINT, never from
  re-supplied CLI flags, so seed / streams / schedule cannot drift;
- the fork records `forked_from = "{source_run_id}/step_NNNNNNN"` in
  config.json and every checkpoint's metadata (Week 9 audit trail);
- Trainer._resume replays the deterministic stream to the fork step, so
  the post-resume batch sequence equals the source run's continuation
  (validated by scripts/validate_resume.py).

Re-running the same command after a kill is safe: if the fork run already
has checkpoints of its own, it resumes from its own latest one (with the
intervention stamp restored from that checkpoint) instead of re-forking.

Examples (Week 4 arith pilot; d50 onset = step 2500):
  python scripts/run_intervention.py --from-run arith_delay50_exp20_size23M_seed1 \\
      --fork-step 2500 --intervention reset
  python scripts/run_intervention.py --from-run arith_delay50_exp20_size23M_seed1 \\
      --fork-step 2500 --intervention reset_spike
  python scripts/run_intervention.py --from-run arith_delay50_exp20_size23M_seed1 \\
      --fork-step 2500 --intervention lr_shift --shift-source-start 0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.checkpoint import checkpoint_path, load_checkpoint  # noqa: E402
from reopen.config import CHECKPOINT_DIR, ModelConfig, TrainConfig  # noqa: E402
from reopen.interventions import (  # noqa: E402
    CramIntervention, FreezeIntervention, LRSpikeIntervention,
    ResetIntervention, ShiftedLRWindow,
)
from reopen.train import Trainer  # noqa: E402

# ---------------------------------------------------------------------------
# Provisional intervention parameters (STANDING RULE: single location, tune
# against the Week 4 pilot design doc before launching the real conditions).
# ---------------------------------------------------------------------------
SPIKE_PEAK_LR_PLACEHOLDER = 6e-4     # = baseline peak lr; TODO: pilot doc
SPIKE_DURATION_PLACEHOLDER = 300     # steps; TODO: pilot doc
CRAM_DENSITY_PLACEHOLDER = 0.45      # just under the 0.5 stream-dominance cap
CRAM_DURATION_PLACEHOLDER = 500      # steps; TODO: pilot doc
FREEZE_DURATION_PLACEHOLDER = 500    # steps; TODO: pilot doc


def fork_config(source_ckpt: dict, iv_suffix: str) -> TrainConfig:
    """Rebuild the source run's TrainConfig from its checkpoint and rename
    it into the fork's namespace. Importable for tests."""
    stored = source_ckpt["train_config"]
    assert stored.get("intervention") is None, (
        f"source checkpoint already carries intervention "
        f"{stored['intervention']!r} — forks must start from a clean run"
    )
    known = {f.name for f in fields(TrainConfig)}
    kwargs = {k: v for k, v in stored.items() if k in known}
    kwargs["betas"] = tuple(kwargs["betas"])
    src = TrainConfig(**kwargs)
    src.forked_from = f"{src.run_id}/step_{source_ckpt['step']:07d}"
    src.run_suffix = (f"{src.run_suffix}_{iv_suffix}" if src.run_suffix
                      else iv_suffix)
    return src


def build_triggers(name: str, args, cfg: TrainConfig) -> tuple[list, str]:
    """(triggers, run_suffix_tag) for the requested intervention."""
    at = args.iv_at
    if name == "reset":
        tgt = args.reset_target.replace("block:", "blk")
        return [ResetIntervention(at_step=at, target=args.reset_target)], \
            f"ivreset{tgt}{at}"
    if name == "spike":
        return [LRSpikeIntervention(
            at_step=at, duration=args.spike_duration,
            peak_lr=args.spike_peak_lr)], f"ivspike{at}"
    if name == "reset_spike":
        # the reset x LR-schedule confound contrast arm: same reset, plus
        # fresh gradient signal for the re-randomized params
        return [
            ResetIntervention(at_step=at, target=args.reset_target),
            LRSpikeIntervention(at_step=at, duration=args.spike_duration,
                                peak_lr=args.spike_peak_lr),
        ], f"ivresetspike{args.reset_target.replace('block:', 'blk')}{at}"
    if name == "freeze":
        trainable = args.freeze_trainable.split(",")
        return [FreezeIntervention(
            at_step=at, unfreeze_step=args.freeze_until,
            trainable=trainable)], f"ivfreeze{at}"
    if name == "cram":
        return [CramIntervention(
            at_step=at, end_step=args.cram_until, stream=cfg.skill,
            boosted_density=args.cram_density)], f"ivcram{at}"
    if name == "lr_shift":
        return [ShiftedLRWindow(
            at_step=at, end_step=args.shift_until,
            source_start=args.shift_source_start)], \
            f"ivlrshift{args.shift_source_start}at{at}"
    if name == "cram_lrshift":
        # Q1 disambiguation arm: cram's density boost UNDER the restored early
        # LR schedule. Separates "cram failed because dose was capped" from
        # "cram failed because the boosted data was LR-starved like everything
        # else at d50". cram ~ lr_shift alone -> LR did all the work;
        # cram >> lr_shift alone -> the extra data was useful but LR-starved.
        return [
            CramIntervention(
                at_step=at, end_step=args.cram_until, stream=cfg.skill,
                boosted_density=args.cram_density),
            ShiftedLRWindow(
                at_step=at, end_step=args.shift_until,
                source_start=args.shift_source_start),
        ], f"ivcramlrshift{args.shift_source_start}at{at}"
    raise ValueError(f"unknown intervention {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True,
                    help="source run_id (must have a checkpoint at "
                         "--fork-step)")
    ap.add_argument("--fork-step", type=int, required=True,
                    help="exact checkpoint step to fork from — the COMMON "
                         "checkpoint shared by every intervention arm")
    ap.add_argument("--intervention", required=True,
                    choices=["reset", "spike", "reset_spike", "freeze",
                             "cram", "lr_shift", "cram_lrshift"])
    ap.add_argument("--iv-at", type=int, default=None,
                    help="step the intervention fires (default: fork+1)")
    ap.add_argument("--run-suffix", default="",
                    help="override the auto-generated fork suffix")
    ap.add_argument("--reset-target", default="embedding",
                    help='"embedding" | "lm_head" | "block:<i>"')
    ap.add_argument("--spike-peak-lr", type=float,
                    default=SPIKE_PEAK_LR_PLACEHOLDER)
    ap.add_argument("--spike-duration", type=int,
                    default=SPIKE_DURATION_PLACEHOLDER)
    ap.add_argument("--freeze-until", type=int, default=None)
    ap.add_argument("--freeze-trainable", default="embedding",
                    help="comma-separated component targets that STAY "
                         "trainable")
    ap.add_argument("--cram-until", type=int, default=None)
    ap.add_argument("--cram-density", type=float,
                    default=CRAM_DENSITY_PLACEHOLDER)
    ap.add_argument("--shift-until", type=int, default=None,
                    help="lr_shift window end (default: total_steps + 1 = "
                         "to the end of the run)")
    ap.add_argument("--shift-source-start", type=int, default=0,
                    help="schedule step the window's first step replays")
    args = ap.parse_args()

    src_path = checkpoint_path(args.from_run, args.fork_step)
    assert src_path.exists(), f"no checkpoint at {src_path}"
    src_ckpt = load_checkpoint(src_path, device="cpu")
    assert src_ckpt["step"] == args.fork_step

    if args.iv_at is None:
        args.iv_at = args.fork_step + 1
    assert args.iv_at > args.fork_step, (
        "intervention step must be AFTER the fork checkpoint — an earlier "
        "step would need weights that checkpoint no longer represents"
    )
    total = src_ckpt["train_config"]["total_steps"]
    if args.freeze_until is None:
        args.freeze_until = args.iv_at + FREEZE_DURATION_PLACEHOLDER
    if args.cram_until is None:
        args.cram_until = args.iv_at + CRAM_DURATION_PLACEHOLDER
    if args.shift_until is None:
        args.shift_until = total + 1  # never hands back; run ends first

    # build once against a scratch config for the suffix, then rebuild the
    # real config with that suffix baked into run_id
    scratch = fork_config(src_ckpt, "scratch")
    triggers, auto_suffix = build_triggers(args.intervention, args, scratch)
    cfg = fork_config(src_ckpt, args.run_suffix or auto_suffix)

    # kill-tolerant relaunch: prefer this fork's own latest checkpoint
    own = sorted((CHECKPOINT_DIR / cfg.run_id).glob("step_*.pt"))
    resume_path = src_path
    if own:
        resume_path = own[-1]
        own_tc = load_checkpoint(resume_path, device="cpu")["train_config"]
        assert own_tc.get("forked_from") == cfg.forked_from, (
            f"existing checkpoints under {cfg.run_id} fork from "
            f"{own_tc.get('forked_from')!r}, not {cfg.forked_from!r}"
        )
        # restore the stamp; triggers that already fired won't refire
        cfg.intervention = own_tc["intervention"]
        cfg.intervention_step = own_tc["intervention_step"]
        print(f"[fork] resuming own checkpoint {resume_path.name} "
              f"(intervention stamp: {cfg.intervention})")
    else:
        print(f"[fork] {cfg.run_id} <- {cfg.forked_from} "
              f"({args.intervention} at step {args.iv_at})")

    model_cfg = ModelConfig(**src_ckpt["model_config"])
    trainer = Trainer(model_cfg, cfg, triggers=triggers,
                      resume_from=resume_path)
    trainer.train()


if __name__ == "__main__":
    # Windows spawn: everything stays under the __main__ guard
    main()
