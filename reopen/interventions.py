"""Intervention module — Tier 1 reopening methods (SCAFFOLD).

Status: structure built in Week 2 per the walkthrough; NOT wired into any
run until the Week 3 pilot confirms a critical period exists. Each
intervention implements the Trainer's Trigger protocol — __call__(step,
trainer) fired at the start of every step — so Week 4 wiring is just
`Trainer(..., triggers=[intervention])`.

The four Tier-1 methods (locked in CLAUDE.md):

1. ResetIntervention    — re-randomize the embedding / a block / the head
                          at an exact step. Biggest forgetting risk: watch
                          general perplexity closely on these runs.
2. LRSpikeIntervention  — boost LR at re-exposure, cosine back down to the
                          scheduled value over `duration` steps.
3. FreezeIntervention   — freeze everything except chosen components for a
                          step window (requires_grad toggling; AdamW skips
                          params with no grad).
4. CramIntervention     — temporarily boost the skill stream's density
                          (control condition). Uses MixedLoader's
                          segmented set_density seam; the extra exposure
                          is visible in tokens_by_stream / metrics, and the
                          equal-budget accounting happens in analysis.

Every intervention stamps (name, step) into trainer.cfg so all subsequent
checkpoints carry it in their metadata dict (locked convention).

Component targets accepted by reset/freeze:
    "embedding" | "lm_head" | "block:<i>" (e.g. "block:3")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from .train import lr_at_step

# Trigger classes whose effect lives OUTSIDE the checkpoint (loader density,
# requires_grad flags) set replay_fire = True: Trainer._resume re-fires them
# during the stream-replay fast-forward so a killed-and-resumed run
# reconstructs both its data stream and its grad state exactly. Triggers
# whose effect IS captured by the checkpoint (weight resets — already in the
# saved weights) or recomputed every in-window step (LR overrides) must NOT
# refire on replay: a replayed reset would re-randomize weights that were
# just loaded.


def _target_params(model, target: str):
    if target == "embedding":
        return list(model.wte.parameters())
    if target == "lm_head":
        return list(model.lm_head.parameters())
    if target.startswith("block:"):
        return list(model.blocks[int(target.split(":")[1])].parameters())
    raise ValueError(f"unknown target {target!r}")


def _clear_optimizer_state(optimizer, params) -> None:
    """Drop Adam moments for re-initialized params — stale first/second
    moment estimates would immediately drag fresh weights back toward the
    old solution, contaminating the reset."""
    for p in params:
        optimizer.state.pop(p, None)


def _stamp(trainer, name: str, step: int) -> None:
    trainer.cfg.intervention = name
    trainer.cfg.intervention_step = step


@dataclass
class ResetIntervention:
    at_step: int
    target: str = "embedding"

    def __call__(self, step: int, trainer) -> None:
        if step != self.at_step:
            return
        model = trainer.raw_model
        if self.target == "embedding":
            model.reset_embedding()
        elif self.target == "lm_head":
            model.reset_lm_head()
        elif self.target.startswith("block:"):
            model.reset_block(int(self.target.split(":")[1]))
        else:
            raise ValueError(f"unknown target {self.target!r}")
        _clear_optimizer_state(trainer.optimizer,
                               _target_params(model, self.target))
        _stamp(trainer, f"reset_{self.target}", step)
        print(f"[intervention] step {step}: reset {self.target} "
              "(+ cleared its optimizer state)")


@dataclass
class LRSpikeIntervention:
    at_step: int
    duration: int
    peak_lr: float

    def __call__(self, step: int, trainer) -> None:
        if step == self.at_step:
            _stamp(trainer, f"lr_spike_{self.peak_lr:g}", step)
            print(f"[intervention] step {step}: LR spike to "
                  f"{self.peak_lr:g} over {self.duration} steps")
        if self.at_step <= step < self.at_step + self.duration:
            sched = lr_at_step(step, trainer.cfg)
            t = (step - self.at_step) / max(1, self.duration)
            w = 0.5 * (1 + math.cos(math.pi * t))  # 1 -> 0
            trainer.lr_override = sched + (self.peak_lr - sched) * w
        elif step == self.at_step + self.duration:
            trainer.lr_override = None  # hand control back to the schedule


@dataclass
class FreezeIntervention:
    at_step: int
    unfreeze_step: int
    trainable: list[str] = field(default_factory=list)  # stays trainable
    # requires_grad flags are not checkpointed — a resume landing inside
    # [at_step, unfreeze_step) must refire the freeze during replay
    replay_fire: ClassVar[bool] = True

    def __call__(self, step: int, trainer) -> None:
        model = trainer.raw_model
        if step == self.at_step:
            keep = [p for t in self.trainable
                    for p in _target_params(model, t)]
            keep_ids = {id(p) for p in keep}
            n_frozen = 0
            for p in model.parameters():
                p.requires_grad = id(p) in keep_ids
                n_frozen += int(not p.requires_grad)
            _stamp(trainer, f"freeze_except_{'+'.join(self.trainable)}", step)
            print(f"[intervention] step {step}: froze {n_frozen} tensors; "
                  f"trainable: {self.trainable}")
        elif step == self.unfreeze_step:
            for p in model.parameters():
                p.requires_grad = True
            print(f"[intervention] step {step}: unfroze all parameters")


@dataclass
class CramIntervention:
    at_step: int
    end_step: int
    stream: str          # e.g. "arith" / "syntax" / "vocab"
    boosted_density: float
    # loader density is not checkpointed, and the resume replay must walk
    # the SAME stream sequence the run actually saw — refire on replay
    replay_fire: ClassVar[bool] = True

    def _orig_density(self, trainer) -> float:
        # Look up the stream's original resolved density from trainer.streams
        # rather than caching it at at_step. Caching in instance state broke
        # on resume: a resume landing between at_step and end_step re-runs
        # at_step via replay_fire (so the cache would be set), BUT if replay
        # is ever skipped the end_step restore referenced an unset attribute
        # and crashed. Deriving it from trainer.streams is resume-robust and
        # order-independent.
        return next(s.density for s in trainer.streams
                    if s.name == self.stream)

    def __call__(self, step: int, trainer) -> None:
        if step == self.at_step:
            orig = self._orig_density(trainer)
            trainer.loader.set_density(self.stream, self.boosted_density)
            _stamp(trainer, f"cram_{self.boosted_density:g}", step)
            print(f"[intervention] step {step}: cram — {self.stream} "
                  f"density {orig:g} -> {self.boosted_density:g}")
        elif step == self.end_step:
            trainer.loader.set_density(self.stream,
                                       self._orig_density(trainer))
            print(f"[intervention] step {step}: cram over — {self.stream} "
                  f"density back to {self._orig_density(trainer):g}")


@dataclass
class ShiftedLRWindow:
    """LR-segment dissection instrument (Week 4 arith): during steps
    [at_step, end_step), follow the LR the schedule would produce at
    (step - at_step + source_start) — i.e. replay a different SEGMENT of
    the same warmup+cosine schedule inside this window.

    The delayed-model use: fork the d50 run at its onset and replay the
    schedule segment the EARLY window enjoyed (source_start=0 → warmup +
    first-half cosine). If the delay gap closes under this alone, the
    "critical period" is optimizer-schedule closure, not representational
    entrenchment; if it persists, LR is exonerated. The complement (early
    window under late-segment LR) uses source_start = the late onset.

    Distinct from LRSpikeIntervention (a transient boost decaying back to
    the CURRENT schedule): this reproduces a full schedule segment,
    warmup ramp included. Do not run both — they fight over lr_override.
    """
    at_step: int
    end_step: int       # exclusive; hands back to the schedule here
    source_start: int   # schedule step the window's first step maps to

    def __call__(self, step: int, trainer) -> None:
        if step == self.at_step:
            _stamp(trainer, f"lr_shift_src{self.source_start}", step)
            print(f"[intervention] step {step}: LR window replays schedule "
                  f"segment [{self.source_start}, "
                  f"{self.source_start + self.end_step - self.at_step})")
        if self.at_step <= step < self.end_step:
            trainer.lr_override = lr_at_step(
                step - self.at_step + self.source_start, trainer.cfg)
        elif step == self.end_step:
            trainer.lr_override = None


@dataclass
class EarlyBurstControl:
    """Density-decoupling control for the Detect phase (NOT a reopening
    intervention): same total skill tokens and window length as a delay
    condition, positioned at the START of training — steps [1, end_step]
    at burst_density, then EXACTLY ZERO afterward (a nonzero tail would be
    'burst plus extra exposure' and reintroduce the confound this control
    exists to remove).

    early-burst ~= matched delay condition  => burst density drives the gap
    early-burst ~= delay-0 baseline         => timing drives it
    """
    stream: str
    burst_density: float
    end_step: int
    replay_fire: ClassVar[bool] = True  # same reason as CramIntervention

    def __call__(self, step: int, trainer) -> None:
        if step == 1:
            trainer.loader.set_density(self.stream, self.burst_density)
            print(f"[control] step 1: early burst — {self.stream} "
                  f"density {self.burst_density:g} until step "
                  f"{self.end_step}")
        elif step == self.end_step + 1:
            trainer.loader.set_density(self.stream, 0.0)
            _stamp(trainer, f"early_burst_end_{self.burst_density:g}", step)
            print(f"[control] step {step}: burst over — {self.stream} "
                  f"density -> 0.0 for the rest of training")
