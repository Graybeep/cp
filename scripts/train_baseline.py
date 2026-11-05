"""CLI entry point for baseline (no-delay, no-intervention) training runs.

Week 2 sanity check:
    python scripts/train_baseline.py --total-steps 1000

All schedule numbers default to the provisional values in reopen/config.py
(STANDING RULE: they are tuned there, not here — CLI flags exist so a run
can override without editing code).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import MODEL_SIZES, ModelConfig, TrainConfig, model_config_for  # noqa: E402
from reopen.train import Trainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    defaults = TrainConfig()
    ap.add_argument("--run-name", default=defaults.run_name)
    ap.add_argument("--model-size", choices=list(MODEL_SIZES),
                    default=defaults.model_size_tag,
                    help="architecture from the config.py size registry; "
                         "sets both the model dims and the run_id size tag")
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--total-steps", type=int, default=defaults.total_steps)
    ap.add_argument("--checkpoint-interval", type=int,
                    default=defaults.checkpoint_interval)
    ap.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    ap.add_argument("--lr", type=float, default=defaults.lr)
    ap.add_argument("--min-lr-ratio", type=float, default=defaults.min_lr_ratio,
                    help="cosine floor = lr * this. Raise (e.g. 0.5) to sustain "
                         "high LR through training (LR-budget schedule-shape test).")
    ap.add_argument("--lr-schedule", choices=["cosine", "wsd"],
                    default=defaults.lr_schedule,
                    help="cosine = locked baseline; wsd = warmup-stable-decay "
                         "(same peak/floor/warmup, constant plateau then a "
                         "linear decay over the final --wsd-decay-frac). "
                         "Non-cosine runs get their own run_id namespace.")
    ap.add_argument("--wsd-decay-frac", type=float,
                    default=defaults.wsd_decay_frac,
                    help="wsd only: final fraction of steps spent decaying")
    ap.add_argument("--micro-batch-size", type=int,
                    default=defaults.micro_batch_size)
    ap.add_argument("--grad-accum-steps", type=int,
                    default=defaults.grad_accum_steps)
    ap.add_argument("--num-workers", type=int, default=defaults.num_workers)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--skill", choices=["none", "arith", "syntax", "vocab"],
                    default="none")
    ap.add_argument("--delay-onset", type=float, default=None,
                    help="fraction of total steps (e.g. 0.5); "
                         "defaults to 0.0 when --skill is set")
    ap.add_argument("--skill-exposure-frac", type=float,
                    default=defaults.skill_exposure_frac)
    ap.add_argument("--run-suffix", default="",
                    help="run_id disambiguator for reruns under changed "
                         "data (e.g. poolv2)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from this run_id's latest checkpoint "
                         "(fresh start if none exists — manual use)")
    ap.add_argument("--burst-end-frac", type=float, default=None,
                    help="early-burst density-decoupling CONTROL: pack the "
                         "full skill exposure into steps [1, frac*total] "
                         "at density exposure/frac, then zero. Mirrors the "
                         "delay condition with onset (1-frac) — same "
                         "tokens, same window length, opposite timing. "
                         "Mutually exclusive with --delay-onset")
    ap.add_argument("--cram", default=None, metavar="AT:END:DENSITY",
                    help="cram intervention (stateful — validates the resume "
                         "replay_fire path via the loader-density seam): "
                         "boost the skill stream's density to DENSITY during "
                         "steps [AT, END), then restore. AT and END are "
                         "FRACTIONS of total steps. Needs --skill")
    ap.add_argument("--freeze", default=None, metavar="AT:UNTIL:TRAINABLE",
                    help="freeze intervention (stateful — validates the "
                         "resume replay_fire path): freeze ALL params except "
                         "TRAINABLE (comma-list of component targets, e.g. "
                         "embedding or block:3) during steps [AT, UNTIL). "
                         "AT and UNTIL are FRACTIONS of total steps")
    ap.add_argument("--lr-shift", default=None, metavar="AT:END:SRC",
                    help="LR-segment dissection (Week 4): during steps "
                         "[AT, END) follow the schedule segment starting "
                         "at SRC instead — all three are FRACTIONS of "
                         "total steps (converted to exact steps once, per "
                         "the standing rule). e.g. 0.0:0.5:0.5 runs the "
                         "early window under the late (decayed) segment")
    ap.add_argument("--resume-strict", action="store_true",
                    help="resume; HARD ERROR if no checkpoint exists. Use "
                         "for supervisor retries — a missing checkpoint on "
                         "a retry means something is wrong (path/flush/"
                         "race), and silently restarting from random init "
                         "under the same run_id would corrupt the run")
    args = ap.parse_args()

    delay = args.delay_onset
    if args.skill != "none" and delay is None:
        delay = 0.0
    cram_spec = None
    if args.cram is not None:
        assert args.skill != "none", "--cram needs --skill"
        at_f, end_f, dens = args.cram.split(":")
        at_f, end_f, dens = float(at_f), float(end_f), float(dens)
        assert 0.0 <= at_f < end_f <= 1.0 and 0.0 < dens < 0.5
        cram_spec = (at_f, end_f, dens)
        if not args.run_suffix:
            args.run_suffix = f"cram{int(round(dens * 100))}"
    freeze_fracs = None
    if args.freeze is not None:
        at_f, until_f, trainable = args.freeze.split(":")
        at_f, until_f = float(at_f), float(until_f)
        assert 0.0 <= at_f < until_f <= 1.0
        freeze_fracs = (at_f, until_f, trainable.split(","))
        if not args.run_suffix:
            args.run_suffix = f"freeze{int(round(at_f * 100))}"
    lr_shift_fracs = None
    if args.lr_shift is not None:
        at_f, end_f, src_f = (float(v) for v in args.lr_shift.split(":"))
        assert 0.0 <= at_f < end_f <= 1.0 and 0.0 <= src_f <= 1.0
        lr_shift_fracs = (at_f, end_f, src_f)
        if not args.run_suffix:
            # an LR-shifted run must not share the plain baseline's run_id
            args.run_suffix = (f"lrshift{int(round(src_f * 100))}"
                               f"at{int(round(at_f * 100))}")
    if args.burst_end_frac is not None:
        assert args.skill != "none" and not args.delay_onset, (
            "--burst-end-frac needs a skill and excludes --delay-onset")
        assert 0.0 < args.burst_end_frac < 1.0
        burst_density = args.skill_exposure_frac / args.burst_end_frac
        assert burst_density < 0.5, (
            f"burst density {burst_density:.2f} would dominate the stream "
            "(same policy cap as resolve_streams)")

    train_cfg = TrainConfig(
        skill=args.skill,
        delay_onset=delay,
        skill_exposure_frac=args.skill_exposure_frac,
        run_name=args.run_name,
        run_suffix=args.run_suffix,
        model_size_tag=args.model_size,
        seed=args.seed,
        total_steps=args.total_steps,
        checkpoint_interval=args.checkpoint_interval,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        lr_schedule=args.lr_schedule,
        wsd_decay_frac=args.wsd_decay_frac,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        compile=args.compile,
    )
    model_cfg = model_config_for(args.model_size)

    resume_path = None
    if args.resume or args.resume_strict:
        from reopen.config import CHECKPOINT_DIR
        hits = sorted((CHECKPOINT_DIR / train_cfg.run_id).glob("step_*.pt"))
        if hits:
            resume_path = hits[-1]
        elif args.resume_strict:
            sys.exit(f"--resume-strict: no checkpoints under "
                     f"{train_cfg.run_id} — refusing to start fresh")
        else:
            print(f"--resume: no checkpoints under {train_cfg.run_id}, "
                  "starting fresh")

    triggers = []
    if args.burst_end_frac is not None:
        from reopen.interventions import EarlyBurstControl
        end_step = int(round(args.burst_end_frac * train_cfg.total_steps))
        triggers.append(EarlyBurstControl(
            stream=args.skill, burst_density=burst_density,
            end_step=end_step,
        ))
        print(f"early-burst control: {args.skill} density "
              f"{burst_density:g} over steps [1, {end_step}], then 0 "
              f"(mirrors delay-{int((1 - args.burst_end_frac) * 100)})")
    if cram_spec is not None:
        from reopen.interventions import CramIntervention
        at_f, end_f, dens = cram_spec
        triggers.append(CramIntervention(
            at_step=max(1, int(round(at_f * train_cfg.total_steps))),
            end_step=int(round(end_f * train_cfg.total_steps)),
            stream=args.skill, boosted_density=dens,
        ))
    if freeze_fracs is not None:
        from reopen.interventions import FreezeIntervention
        at_f, until_f, trainable = freeze_fracs
        triggers.append(FreezeIntervention(
            at_step=max(1, int(round(at_f * train_cfg.total_steps))),
            unfreeze_step=int(round(until_f * train_cfg.total_steps)),
            trainable=trainable,
        ))
    if lr_shift_fracs is not None:
        from reopen.interventions import ShiftedLRWindow
        at_f, end_f, src_f = lr_shift_fracs
        triggers.append(ShiftedLRWindow(
            at_step=max(1, int(round(at_f * train_cfg.total_steps))),
            end_step=int(round(end_f * train_cfg.total_steps)),
            source_start=int(round(src_f * train_cfg.total_steps)),
        ))

    # TEST-ONLY power control for validate_intervention_resume.py: force the
    # replay_fire flag off to simulate the pre-fix bug (stateful triggers
    # NOT re-fired during resume replay). The validation confirms this makes
    # kill+resume diverge from the noise floor — i.e. that the check has the
    # power to detect a broken replay. Never set in production.
    if os.environ.get("REOPEN_BREAK_REPLAY"):
        for t in triggers:
            t.replay_fire = False
        print("[TEST] REOPEN_BREAK_REPLAY set — replay_fire forced OFF")

    trainer = Trainer(model_cfg, train_cfg, triggers=triggers,
                      resume_from=resume_path)
    n = trainer.raw_model.num_params()
    print(f"run_id: {train_cfg.run_id}")
    print(f"model: {n/1e6:.1f}M params "
          f"({model_cfg.n_layer}L, {model_cfg.n_embd}d, {model_cfg.n_head}h, "
          f"vocab {model_cfg.vocab_size}, ctx {model_cfg.block_size})")
    trainer.train()


if __name__ == "__main__":
    # Required on Windows: DataLoader workers use spawn, which re-imports
    # this module — everything must stay under the __main__ guard.
    main()
