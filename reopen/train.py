"""Training loop.

Design points from the locked specs:
- bf16 autocast, gradient accumulation, optional torch.compile
- AdamW + linear warmup + cosine decay, computed per step in Python (not a
  torch scheduler object) so the Week-4 LR-spike intervention can override
  the LR at an exact step through a trigger
- triggers: callables fired at the START of every step with (step, trainer).
  This is the seam where the intervention module (reset / LR-spike /
  freeze / cram) plugs in later — the loop itself never needs to change.
- checkpoints on a fixed step interval (never wall-clock), and every
  checkpoint eval logs BOTH general val loss/perplexity AND skill-accuracy
  columns into one pandas DataFrame. Skill columns are NaN until the skill
  eval harness (build priority #4) fills them in — the pairing exists in
  the schema from day one.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset

import os

from .checkpoint import load_checkpoint, save_checkpoint
from .config import CHECKPOINT_DIR, LOG_DIR, ModelConfig, TrainConfig, save_configs
from .data import TokenBlockDataset, load_meta
from .evals import EvalHarness
from .model import GPT
from .schedule import MixedLoader, exposure_plan, resolve_streams


class Trigger(Protocol):
    """Intervention seam: called at the start of every training step."""

    def __call__(self, step: int, trainer: "Trainer") -> None: ...


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to lr * min_lr_ratio."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * progress))


METRIC_COLUMNS = [
    "run_id", "step", "tokens_seen", "skill_tokens_seen", "lr", "train_loss",
    "val_loss", "val_ppl",
    # Skill evals (build priority #4). syntax_accuracy is the wh-licensing
    # INTERACTION (chance = 0.5 even for a frequency-biased model), not a
    # raw BLiMP paradigm score — see reopen/evals.py.
    "arith_accuracy", "syntax_accuracy", "syntax_accuracy_local",
    "vocab_control_accuracy",
]


class Trainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        triggers: list[Trigger] | None = None,
        resume_from: Path | None = None,
    ):
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.triggers = triggers or []
        self.device = train_cfg.device
        assert train_cfg.warmup_steps < train_cfg.total_steps, (
            f"warmup_steps ({train_cfg.warmup_steps}) must be < total_steps "
            f"({train_cfg.total_steps}) — peak LR would never be reached"
        )

        torch.manual_seed(train_cfg.seed)
        torch.cuda.manual_seed_all(train_cfg.seed)

        tok_dir = Path(train_cfg.tokenized_dir)
        meta = load_meta(tok_dir)
        model_cfg.vocab_size = meta["vocab_size"]

        self.model = GPT(model_cfg).to(self.device)
        self.raw_model = self.model  # pre-compile handle (for saving/resets)
        if train_cfg.compile:
            self.model = torch.compile(self.model)

        self.optimizer = self.raw_model.configure_optimizer(
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=train_cfg.betas,
            device=self.device,
        )
        # Triggers may set this to override the schedule at an exact step
        # (the LR-spike intervention). None = follow the cosine schedule.
        self.lr_override: float | None = None

        # Delay-scheduled training stream (base + optional delayed skill).
        # The exact plan (onset steps, densities, planned token budgets)
        # is resolved up front and written into config.json.
        self.streams = resolve_streams(train_cfg)
        self.plan = exposure_plan(self.streams, train_cfg, model_cfg)
        self.loader = MixedLoader(
            self.streams, model_cfg.block_size, train_cfg.micro_batch_size,
            train_cfg.num_workers, train_cfg.seed,
        )
        self.tokens_by_stream = {s.name: 0 for s in self.streams}
        # Fixed eval subset: evenly-spaced blocks spanning the WHOLE dev
        # split (it is ordered by source corpus, so reading from the front
        # would eval on child-directed speech only). Deterministic across
        # runs and checkpoints — comparable ppl numbers everywhere.
        val_ds = TokenBlockDataset(tok_dir / "val.bin", model_cfg.block_size)
        n_eval = train_cfg.eval_batches * train_cfg.micro_batch_size
        idx = torch.linspace(0, len(val_ds) - 1, n_eval).long().tolist()
        self.val_loader = DataLoader(
            Subset(val_ds, idx),
            batch_size=train_cfg.micro_batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )

        # Skill evals at every checkpoint, from day one (build priority #4)
        self.eval_harness = EvalHarness(
            device=self.device, n_per_skill=train_cfg.skill_eval_n
        )

        self.metrics = pd.DataFrame(columns=METRIC_COLUMNS)
        self.log_dir = LOG_DIR / train_cfg.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        save_configs(self.log_dir / "config.json", model_cfg, train_cfg,
                     extra={"exposure_plan": self.plan})

        # Resume path (Week 4 common-checkpoint rule + supervisor
        # crash/kill tolerance). Checkpoints stored everything from day
        # one; this is the loading side.
        self.start_step = 1
        if resume_from is not None:
            self._resume(resume_from)

    # ------------------------------------------------------------------
    def _resume(self, path: Path) -> None:
        """Restore model/optimizer/step from a checkpoint and fast-forward
        the deterministic data stream to it.

        The stream is replayed (CPU-only, no forward passes) so the
        post-resume batch sequence is IDENTICAL to what an uninterrupted
        run would have seen — required both for reproducibility and for
        the Week-4 rule that intervention runs resuming from a common
        checkpoint must share their post-resume stream. Costs data-loading
        time only (~1-2 min per 5000 steps).

        Note: the train_loss column restarts its EMA at the resume step
        (the EMA itself is not checkpointed); val/skill metrics are
        unaffected.
        """
        ckpt = load_checkpoint(path, device="cpu")
        assert ckpt["train_config"]["total_steps"] == self.cfg.total_steps, (
            "resume with a different total_steps would change the LR "
            "schedule and delay-onset conversion mid-run"
        )
        self.raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        step0 = ckpt["step"]
        self.start_step = step0 + 1

        # Triggers whose effects live outside the checkpoint (loader
        # density, requires_grad flags) refire during replay so the
        # reconstructed stream and grad state match what the run actually
        # saw; see the replay_fire note in reopen/interventions.py.
        replay_trigs = [t for t in self.triggers
                        if getattr(t, "replay_fire", False)]
        for step in range(1, step0 + 1):
            for trig in replay_trigs:
                trig(step, self)
            for _ in range(self.cfg.grad_accum_steps):
                x, _y, stream = self.loader.next_batch(step)
                self.tokens_by_stream[stream] += x.numel()

        mpath = self.log_dir / "metrics.csv"
        if mpath.exists():
            # POWER-LOSS TOLERANCE: metrics.csv is a sidecar written outside
            # the atomic checkpoint path, so a power cut (no fsync) can leave
            # it truncated/blank — a column-less or unparseable frame. That
            # must NOT strand a run whose checkpoint loaded fine (it did, we
            # got here): warn and keep the fresh METRIC_COLUMNS frame from
            # __init__. Historical rows are still in eval_details.jsonl.
            try:
                df = pd.read_csv(mpath)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                df = None
            if df is not None and "step" in df.columns:
                self.metrics = df[df.step <= step0].reset_index(drop=True)
            else:
                print(f"WARNING: {mpath} is corrupt/blank (no 'step' column "
                      "— likely a power-loss partial write); starting the "
                      "metrics log fresh, past eval rows remain in "
                      "eval_details.jsonl")
        epath = self.log_dir / "eval_details.jsonl"
        if epath.exists():
            kept = [l for l in epath.read_text(encoding="utf-8").splitlines()
                    if l.strip() and json.loads(l)["step"] <= step0]
            epath.write_text("\n".join(kept) + ("\n" if kept else ""),
                             encoding="utf-8")
        print(f"resumed {self.cfg.run_id} from step {step0} "
              f"(stream fast-forwarded, {len(self.metrics)} metric rows kept)")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_val(self) -> tuple[float, float]:
        self.model.eval()
        losses = []
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.cfg.eval_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        val_loss = sum(losses) / len(losses)
        return val_loss, math.exp(val_loss)

    def log_checkpoint_metrics(self, step: int, train_loss: float) -> None:
        val_loss, val_ppl = self.eval_val()
        skill_res = self.eval_harness.evaluate(self.model)
        self.model.train()  # harness leaves the model in eval mode
        tokens_per_step = (
            self.cfg.micro_batch_size
            * self.cfg.grad_accum_steps
            * self.model_cfg.block_size
        )
        skill_tokens = sum(
            n for name, n in self.tokens_by_stream.items() if name != "base"
        )

        def _f(v):
            return float("nan") if v is None else v

        row = {
            "run_id": self.cfg.run_id,
            "step": step,
            "tokens_seen": step * tokens_per_step,
            "skill_tokens_seen": skill_tokens,
            "lr": self.current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "arith_accuracy": _f(skill_res["arith_accuracy"]),
            "syntax_accuracy": _f(skill_res["syntax_accuracy"]),
            "syntax_accuracy_local": _f(skill_res["syntax_accuracy_local"]),
            "vocab_control_accuracy": _f(skill_res["vocab_control_accuracy"]),
        }
        self.metrics.loc[len(self.metrics)] = row
        self.metrics.to_csv(self.log_dir / "metrics.csv", index=False)
        # full detail (per-paradigm BLiMP, vocab logprob) as one JSON
        # line per checkpoint
        with open(self.log_dir / "eval_details.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps({
                "step": step, "val_loss": val_loss, "val_ppl": val_ppl,
                **skill_res,
            }) + "\n")

    # ------------------------------------------------------------------
    def train(self) -> pd.DataFrame:
        cfg = self.cfg
        self.model.train()
        ema_loss: float | None = None
        t0 = time.time()
        pbar = tqdm(range(self.start_step, cfg.total_steps + 1),
                    desc=cfg.run_id, ncols=100)

        for step in pbar:
            for trig in self.triggers:
                trig(step, self)

            self.current_lr = (
                self.lr_override if self.lr_override is not None
                else lr_at_step(step, cfg)
            )
            for group in self.optimizer.param_groups:
                group["lr"] = self.current_lr

            self.optimizer.zero_grad(set_to_none=True)
            micro_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y, stream = self.loader.next_batch(step)
                self.tokens_by_stream[stream] += x.numel()
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                (loss / cfg.grad_accum_steps).backward()
                micro_loss += loss.item() / cfg.grad_accum_steps

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            ema_loss = (
                micro_loss if ema_loss is None
                else 0.98 * ema_loss + 0.02 * micro_loss
            )
            pbar.set_postfix(loss=f"{ema_loss:.3f}", lr=f"{self.current_lr:.1e}")

            if step % cfg.checkpoint_interval == 0 or step == cfg.total_steps:
                self.log_checkpoint_metrics(step, ema_loss)
                save_checkpoint(
                    self.raw_model, self.optimizer, step,
                    self.model_cfg, cfg,
                )

        elapsed = time.time() - t0
        # Realized exposure next to the plan — the matched-total-exposure
        # audit for the Detect phase is a file diff, not a re-derivation.
        (self.log_dir / "realized_exposure.json").write_text(json.dumps(
            {"realized_tokens_by_stream": self.tokens_by_stream,
             "planned": self.plan["streams"]}, indent=2,
        ))

        # CHECKPOINT RETENTION — completion-only. This runs solely because the
        # loop above reached total_steps; a supervisor-timeout / crash / power
        # cut mid-run never gets here, so ALL intermediate checkpoints survive
        # for resume (a run never restarts from the beginning). Only once the
        # run is genuinely finished do we thin it to the fork-safe onset set
        # {25/50/75%, final}. Set REOPEN_NO_PRUNE=1 to keep every checkpoint.
        if os.environ.get("REOPEN_NO_PRUNE") != "1":
            from .prune import onset_keep_steps, prune_run
            keep = onset_keep_steps(cfg.total_steps)
            n, b = prune_run(CHECKPOINT_DIR / cfg.run_id,
                             keep_steps=keep, apply=True)
            if n:
                print(f"retention: pruned {n} intermediate checkpoints "
                      f"({b / 1024**3:.1f} GB), kept steps {keep}")
        steps_run = cfg.total_steps - self.start_step + 1
        print(f"done: {cfg.total_steps} steps in {elapsed/60:.1f} min "
              f"({steps_run / elapsed:.2f} steps/s"
              + (f", {steps_run} this launch" if self.start_step > 1 else "")
              + ")")
        print(f"tokens by stream: {self.tokens_by_stream}")
        return self.metrics
