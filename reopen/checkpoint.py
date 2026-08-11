"""Checkpoint save/load implementing the locked convention:

    checkpoints/{run_id}/step_{step_number:07d}.pt

Every file stores: model weights, optimizer state, current step, both
configs, and the metadata dict (skill, delay_onset, model_size, seed,
intervention + intervention_step) — so the Week-9 statistical analysis
never has to re-derive run configs from filenames, and the intervention
module can resume from any checkpoint.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import torch

from .config import CHECKPOINT_DIR, ModelConfig, TrainConfig


def checkpoint_path(run_id: str, step: int, base_dir: Path = CHECKPOINT_DIR) -> Path:
    return base_dir / run_id / f"step_{step:07d}.pt"


def _clean_state_dict(model: torch.nn.Module) -> dict:
    """Strip torch.compile's _orig_mod. prefix so checkpoints load
    identically whether or not the run was compiled."""
    sd = model.state_dict()
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    base_dir: Path = CHECKPOINT_DIR,
) -> Path:
    path = checkpoint_path(train_cfg.run_id, step, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ATOMIC WRITE (supervisor safety): the launch-timeout kill terminates
    # training abruptly and CAN land mid-torch.save, leaving a truncated,
    # unreadable .pt that latest_step() still counts — the resume then picks
    # it and crashes ("failed finding central directory"). Write to a .tmp
    # (NOT matched by the step_*.pt glob) and os.replace() it into place:
    # rename is atomic, so a kill leaves either the previous complete
    # checkpoint or the new complete one, never a partial file.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "step": step,
            "model": _clean_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "metadata": train_cfg.metadata(),
        },
        tmp,
    )
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)
