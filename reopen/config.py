"""Single source of truth for all run configuration.

STANDING RULE — NO PREMATURE HARDCODING:
Checkpoint interval, total training steps, and anything derived from them
(delay-onset steps, intervention trigger steps) are PROVISIONAL until the
Week 2 sanity run gives real throughput numbers. They live here and nowhere
else; every other module receives them through these dataclasses. Change
them here, and only here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Provisional values — tune after the Week 2 sanity run
# --------------------------------------------------------------------------
TOTAL_STEPS_PLACEHOLDER = 5000  # TODO: tune after Week 2 sanity run
CHECKPOINT_INTERVAL_PLACEHOLDER = 100  # TODO: tune after Week 2 sanity run
#   (target: 40-100 checkpoints per run once total steps are known)
WARMUP_STEPS_PLACEHOLDER = 300  # TODO: revisit alongside total steps
SKILL_EXPOSURE_FRAC_PLACEHOLDER = 0.05  # TODO: tune with Week 3 pilot
#   (fraction of all training tokens that are skill tokens at delay 0;
#   delayed runs are automatically densified to match total exposure)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"


@dataclass
class ModelConfig:
    # Locked architecture (CLAUDE.md): 10-30M params, 4-6 layers, hidden
    # 256-512, 8 heads, seq len 256-512, RoPE, pre-norm, GELU, UNTIED
    # embedding/output weights.
    vocab_size: int = 16384  # overwritten from tokenizer meta.json at load
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    block_size: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


# Model-size registry (single source of truth for the Weeks 5-7 size axis and
# the syntax size-sweep). Tag -> architecture overrides; "23M" MUST equal the
# ModelConfig defaults so size23M sweep runs are identical to prior 23M runs.
# 60M extends past the locked 30M ceiling for the syntax acquisition-rate
# sub-study only (VRAM-verified: 2.95 GB peak on the 8 GB 4060).
MODEL_SIZES: dict[str, dict[str, int]] = {
    "11M": {"n_layer": 4, "n_embd": 256, "n_head": 8},   # ~11.5M params
    "23M": {"n_layer": 6, "n_embd": 384, "n_head": 8},   # ~23.2M (default)
    "60M": {"n_layer": 8, "n_embd": 640, "n_head": 10},  # ~60.3M
}


def model_config_for(tag: str, **overrides) -> "ModelConfig":
    """Build a ModelConfig for a size tag; other fields keep their defaults."""
    if tag not in MODEL_SIZES:
        raise ValueError(f"unknown model size tag {tag!r}; "
                         f"known: {sorted(MODEL_SIZES)}")
    return ModelConfig(**MODEL_SIZES[tag], **overrides)


@dataclass
class TrainConfig:
    # ---- run identity (encoded into run_id, checkpoint metadata, logs) ----
    skill: str = "none"          # "arith" / "syntax" / "vocab" once skills exist
    delay_onset: float | None = None  # fraction of total steps; None = no delay
    skill_exposure_frac: float = SKILL_EXPOSURE_FRAC_PLACEHOLDER
    # set by the intervention module when one fires (Week 4); flows into
    # every subsequent checkpoint's metadata dict
    intervention: str | None = None
    intervention_step: int | None = None
    # set by run_intervention.py on forked runs: "{source_run_id}/step_NNNNNNN"
    # — the common-checkpoint audit trail (Week 4 rule: every intervention
    # run must record the exact checkpoint file it resumed from)
    forked_from: str | None = None
    model_size_tag: str = "23M"
    seed: int = 1
    run_name: str = "sanity_babylm"  # human-readable prefix for run_id
    run_suffix: str = ""  # disambiguator for reruns under changed DATA
    #   (e.g. "poolv2" after a skill-pool rebuild): eval numbers are only
    #   comparable between runs trained on the same skill .bin files, so
    #   those runs must not share a run_id namespace

    # ---- schedule (provisional — see STANDING RULE above) ----
    total_steps: int = TOTAL_STEPS_PLACEHOLDER
    checkpoint_interval: int = CHECKPOINT_INTERVAL_PLACEHOLDER
    warmup_steps: int = WARMUP_STEPS_PLACEHOLDER

    # ---- optimizer: locked baseline = AdamW, warmup + cosine decay ----
    lr: float = 6e-4
    min_lr_ratio: float = 0.1    # cosine floor = lr * min_lr_ratio
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # ---- batching (RTX 4060 8GB: micro-batch 4-8, grad accumulation) ----
    micro_batch_size: int = 8
    grad_accum_steps: int = 4    # effective batch = 32 seqs = 16K tokens/step

    # ---- engineering ----
    compile: bool = False        # torch.compile — needs working Triton on Windows
    num_workers: int = 8
    device: str = "cuda"
    eval_batches: int = 32       # val batches per checkpoint eval
    skill_eval_n: int = 500      # instances per skill at checkpoint evals
    #   (deterministic first-N subset; full sets via eval_checkpoints.py)

    # ---- data ----
    tokenized_dir: str = str(DATA_DIR / "tokenized")

    @property
    def run_id(self) -> str:
        """e.g. arith_delay50_exp5_size20M_seed1 (locked naming convention;
        exp = skill exposure %, present on skill runs only — added once
        exposure frac became a swept variable, so runs differing only in
        exposure don't collide in checkpoints/ and logs/)."""
        parts = [self.run_name if self.skill == "none" else self.skill]
        if self.delay_onset is not None:
            parts.append(f"delay{int(self.delay_onset * 100)}")
        if self.skill != "none":
            parts.append(f"exp{int(round(self.skill_exposure_frac * 100))}")
        if self.total_steps != TOTAL_STEPS_PLACEHOLDER:
            # non-default durations get their own namespace so a longer run
            # can't clobber the default-length run's checkpoints
            parts.append(f"steps{self.total_steps}")
        parts.append(f"size{self.model_size_tag}")
        parts.append(f"seed{self.seed}")
        if self.run_suffix:
            parts.append(self.run_suffix)
        return "_".join(parts)

    def delay_onset_step(self) -> int | None:
        """Convert delay-onset fraction to an exact step BEFORE training
        starts (locked convention: triggers fire on exact step count)."""
        if self.delay_onset is None:
            return None
        return int(round(self.delay_onset * self.total_steps))

    def metadata(self) -> dict:
        """The metadata dict stored in every checkpoint (locked convention);
        intervention fields stay None until an intervention fires."""
        return {
            "skill": self.skill,
            "delay_onset": self.delay_onset,
            "skill_exposure_frac": self.skill_exposure_frac,
            "model_size": self.model_size_tag,
            "seed": self.seed,
            "intervention": self.intervention,
            "intervention_step": self.intervention_step,
            "forked_from": self.forked_from,
        }


def save_configs(path: Path, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": asdict(model_cfg), "train": asdict(train_cfg)}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
