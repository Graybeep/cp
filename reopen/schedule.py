"""Delay scheduler (build priority #3).

Converts (skill, delay_onset fraction, skill_exposure_frac) into an exact,
deterministic plan BEFORE training starts, per the locked conventions:

- onset is an exact step number (fraction * total_steps, rounded once),
  never wall-clock;
- total skill-token exposure is MATCHED across delay conditions by
  construction: a stream delayed to fraction d runs at density
  exposure_frac / (1 - d) over the remaining (1 - d) of training, so
  every condition sees the same number of skill tokens AND the same
  number of base tokens — only timing differs;
- stream selection uses error-diffusion (an accumulator emits a skill
  micro-batch each time it crosses 1), not random sampling: realized
  exposure matches the plan to within one micro-batch, skill batches are
  evenly spaced, and the whole schedule is bit-reproducible with no RNG.

Stream sources per skill:
    arith:  base = tokenized/train.bin (full BabyLM; no arithmetic there)
            skill = skills/arith/train.bin
    syntax: base = skills/syntax/base_nosyntax.bin (wh-object-gap lines
            removed — they must be absent from the early stream)
            skill = skills/syntax/syntax.bin (the held-out lines)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR, ModelConfig, TrainConfig
from .data import cycle, make_loader


@dataclass(frozen=True)
class StreamSpec:
    name: str          # "base" | "arith" | "syntax"
    bin_path: Path
    onset_step: int    # completed steps before exposure: active when
    #                    step > onset_step (Trainer steps run 1..total)
    density: float     # fraction of micro-batches once active (base: rest)


def resolve_streams(cfg: TrainConfig) -> list[StreamSpec]:
    """Compute the exact stream plan for a run. First entry is always the
    base stream; any further entries are delayed skill streams."""
    tok_dir = Path(cfg.tokenized_dir)
    skills_dir = DATA_DIR / "skills"

    if cfg.skill == "none":
        return [StreamSpec("base", tok_dir / "train.bin", 0, 1.0)]

    d = cfg.delay_onset or 0.0
    assert 0.0 <= d < 1.0, f"delay_onset must be in [0, 1), got {d}"
    onset = cfg.delay_onset_step() or 0
    density = cfg.skill_exposure_frac / (1.0 - d)
    assert density < 0.5, (
        f"skill density {density:.2f} (exposure {cfg.skill_exposure_frac} "
        f"at delay {d}) would dominate the stream — lower exposure_frac "
        f"or delay"
    )

    if cfg.skill == "arith":
        base_bin = tok_dir / "train.bin"
        skill_bin = skills_dir / "arith" / "train.bin"
    elif cfg.skill == "syntax":
        base_bin = skills_dir / "syntax" / "base_nosyntax.bin"
        skill_bin = skills_dir / "syntax" / "syntax.bin"
    elif cfg.skill == "vocab":
        # negative control: expected to show NO delay-dependent gap
        base_bin = skills_dir / "vocab" / "base_novocab.bin"
        skill_bin = skills_dir / "vocab" / "vocab.bin"
    else:
        raise ValueError(f"unknown skill {cfg.skill!r}")
    for p in (base_bin, skill_bin):
        assert p.exists(), f"missing {p} — run scripts/prepare_skills.py"

    return [
        StreamSpec("base", base_bin, 0, 1.0 - density),
        StreamSpec(cfg.skill, skill_bin, onset, density),
    ]


def exposure_plan(specs: list[StreamSpec], cfg: TrainConfig,
                  model_cfg: ModelConfig) -> dict:
    """The planned token budget, written into the run's config.json so
    matched exposure across conditions can be audited without re-deriving
    anything (and checked against realized counts in metrics.csv)."""
    tokens_per_micro = cfg.micro_batch_size * model_cfg.block_size
    total_micro = cfg.total_steps * cfg.grad_accum_steps
    plan: dict = {
        "tokens_per_micro_batch": tokens_per_micro,
        "total_micro_batches": total_micro,
        "total_tokens": total_micro * tokens_per_micro,
        "streams": {},
    }
    skill_micro = 0
    for s in specs[1:]:
        active = (cfg.total_steps - s.onset_step) * cfg.grad_accum_steps
        n = int(active * s.density)  # error diffusion realizes exactly this
        skill_micro += n
        plan["streams"][s.name] = {
            "onset_step": s.onset_step,
            "density": s.density,
            "planned_micro_batches": n,
            "planned_tokens": n * tokens_per_micro,
        }
    plan["streams"]["base"] = {
        "onset_step": 0,
        "planned_micro_batches": total_micro - skill_micro,
        "planned_tokens": (total_micro - skill_micro) * tokens_per_micro,
    }
    return plan


class MixedLoader:
    """Yields micro-batches drawn from the stream plan.

    next_batch(step) -> (x, y, stream_name). Stream choice is by exact
    step count and error diffusion only — deterministic and reproducible.
    Within each stream, batches come from a shuffled DataLoader seeded by
    (cfg seed + stream index), cycling when a small stream is exhausted.
    """

    def __init__(self, specs: list[StreamSpec], block_size: int,
                 micro_batch_size: int, num_workers: int, seed: int):
        self.specs = specs
        self.delayed = specs[1:]
        # Emission bookkeeping is multiplicative, not an additive
        # accumulator: repeated float addition drifts (10 x 0.1 < 1.0) and
        # silently drops emissions; floor(calls * density) cannot.
        # Bookkeeping is SEGMENTED so density can change mid-run (the cram
        # intervention): each segment counts from its own (calls, emitted)
        # snapshot, so a density change never applies retroactively.
        self.active_calls = {s.name: 0 for s in self.delayed}
        self.emitted = {s.name: 0 for s in self.delayed}
        self._segment = {s.name: (0, 0, s.density) for s in self.delayed}
        self.iters = {}
        for i, s in enumerate(specs):
            # base gets the full worker budget; small skill bins live in
            # page cache and need at most a couple of workers
            workers = num_workers if s.name == "base" else min(2, num_workers)
            self.iters[s.name] = iter(cycle(make_loader(
                s.bin_path, block_size, micro_batch_size,
                num_workers=workers, seed=seed + i,
            )))

    def set_density(self, name: str, density: float) -> None:
        """Change a delayed stream's density from now on (cram
        intervention seam). Starts a new bookkeeping segment; past
        emissions are untouched."""
        assert name in self.active_calls, f"unknown stream {name!r}"
        assert 0.0 <= density < 1.0
        self._segment[name] = (
            self.active_calls[name], self.emitted[name], density,
        )

    def next_batch(self, step: int):
        chosen = None
        best_deficit = 0
        for s in self.delayed:
            if step > s.onset_step:
                self.active_calls[s.name] += 1
                calls0, emitted0, density = self._segment[s.name]
                target = emitted0 + int(
                    (self.active_calls[s.name] - calls0) * density + 1e-9
                )
                deficit = target - self.emitted[s.name]
                if deficit > best_deficit:
                    best_deficit = deficit
                    chosen = s
        if chosen is not None:
            self.emitted[chosen.name] += 1
            name = chosen.name
        else:
            name = "base"
        x, y = next(self.iters[name])
        return x, y, name
