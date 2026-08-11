"""Unit tests for the intervention scaffold (plain asserts).

Run:  python scripts/test_interventions.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reopen.config import ModelConfig, TrainConfig  # noqa: E402
from reopen.interventions import (  # noqa: E402
    CramIntervention, EarlyBurstControl, FreezeIntervention,
    LRSpikeIntervention, ResetIntervention, ShiftedLRWindow,
)
from reopen.model import GPT  # noqa: E402
from reopen.schedule import MixedLoader, StreamSpec  # noqa: E402
from reopen.train import lr_at_step  # noqa: E402


def _tiny_trainer() -> SimpleNamespace:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_head=8, n_embd=64,
                      block_size=32)
    model = GPT(cfg)
    tcfg = TrainConfig(run_name="ivtest", total_steps=100, warmup_steps=10)
    opt = model.configure_optimizer(tcfg.lr, tcfg.weight_decay, tcfg.betas,
                                    device="cpu")
    # populate optimizer state with one real step
    x = torch.randint(0, 256, (2, 32))
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return SimpleNamespace(raw_model=model, optimizer=opt, cfg=tcfg,
                           lr_override=None, streams=None, loader=None)


def test_reset() -> None:
    tr = _tiny_trainer()
    model = tr.raw_model
    emb_before = model.wte.weight.clone()
    head_before = model.lm_head.weight.clone()
    assert model.wte.weight in tr.optimizer.state, "state not populated"

    iv = ResetIntervention(at_step=5, target="embedding")
    for step in (4, 6):
        iv(step, tr)
    assert torch.equal(model.wte.weight, emb_before), "fired off-step"
    iv(5, tr)
    assert not torch.equal(model.wte.weight, emb_before), "did not reset"
    assert torch.equal(model.lm_head.weight, head_before), \
        "reset must not touch lm_head"
    assert model.wte.weight not in tr.optimizer.state, \
        "stale Adam moments must be cleared for reset params"
    assert model.lm_head.weight in tr.optimizer.state, \
        "other params keep their optimizer state"
    assert tr.cfg.intervention == "reset_embedding"
    assert tr.cfg.intervention_step == 5
    print("reset OK")


def test_lr_spike() -> None:
    tr = _tiny_trainer()
    iv = LRSpikeIntervention(at_step=50, duration=10, peak_lr=5e-3)
    lrs = {}
    for step in range(1, 101):
        iv(step, tr)
        lrs[step] = tr.lr_override if tr.lr_override is not None \
            else lr_at_step(step, tr.cfg)
    assert abs(lrs[50] - 5e-3) < 1e-9, "spike start must hit peak_lr"
    assert lrs[49] == lr_at_step(49, tr.cfg), "no spike before window"
    assert lrs[60] == lr_at_step(60, tr.cfg), "must hand back to schedule"
    assert tr.lr_override is None
    assert all(lrs[s] > lr_at_step(s, tr.cfg) for s in range(50, 59)), \
        "boosted throughout the window"
    assert all(lrs[s] >= lrs[s + 1] for s in range(50, 60)), \
        "cosine decay back down"
    print("lr spike OK")


def test_freeze() -> None:
    tr = _tiny_trainer()
    model = tr.raw_model
    iv = FreezeIntervention(at_step=10, unfreeze_step=20,
                            trainable=["block:1"])
    iv(10, tr)
    assert not model.wte.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    assert not any(p.requires_grad for p in model.blocks[0].parameters())
    assert all(p.requires_grad for p in model.blocks[1].parameters())
    assert tr.cfg.intervention == "freeze_except_block:1"
    iv(15, tr)  # no-op mid-window
    assert not model.wte.weight.requires_grad
    iv(20, tr)
    assert all(p.requires_grad for p in model.parameters())
    print("freeze OK")


def test_cram() -> None:
    with tempfile.TemporaryDirectory() as td:
        base_bin = Path(td) / "base.bin"
        skill_bin = Path(td) / "skill.bin"
        np.full(20_000, 1, dtype=np.uint16).tofile(base_bin)
        np.full(20_000, 2, dtype=np.uint16).tofile(skill_bin)
        specs = [
            StreamSpec("base", base_bin, 0, 0.9),
            StreamSpec("arith", skill_bin, 0, 0.1),
        ]
        loader = MixedLoader(specs, block_size=8, micro_batch_size=2,
                             num_workers=0, seed=1)
        tr = SimpleNamespace(streams=specs, loader=loader,
                             cfg=TrainConfig(run_name="cramtest"))
        iv = CramIntervention(at_step=51, end_step=101, stream="arith",
                              boosted_density=0.5)
        counts = {"pre": 0, "cram": 0, "post": 0}
        for step in range(1, 151):
            iv(step, tr)
            _, _, name = loader.next_batch(step)
            if name == "arith":
                phase = "pre" if step <= 50 else \
                    "cram" if step <= 100 else "post"
                counts[phase] += 1
        assert counts["pre"] == 5, counts    # 50 calls * 0.1
        assert counts["cram"] == 25, counts  # 50 calls * 0.5
        assert counts["post"] == 5, counts   # 50 calls * 0.1 (restored)
        # release memmap handles so Windows can delete the temp dir
        import gc
        del loader, tr
        gc.collect()
    print("cram OK (segmented density bookkeeping exact)")


def test_shifted_lr_window() -> None:
    tr = _tiny_trainer()
    iv = ShiftedLRWindow(at_step=50, end_step=80, source_start=0)
    lrs = {}
    for step in range(1, 101):
        iv(step, tr)
        lrs[step] = tr.lr_override if tr.lr_override is not None \
            else lr_at_step(step, tr.cfg)
    assert lrs[49] == lr_at_step(49, tr.cfg), "no shift before window"
    assert lrs[50] == lr_at_step(0, tr.cfg), \
        "window start must replay the source segment's first step"
    assert lrs[79] == lr_at_step(29, tr.cfg), "window replays source+offset"
    assert lrs[80] == lr_at_step(80, tr.cfg), "must hand back to schedule"
    assert tr.lr_override is None
    assert tr.cfg.intervention == "lr_shift_src0"
    print("shifted lr window OK")


def test_replay_fire_markers() -> None:
    # effects living outside the checkpoint refire on resume replay;
    # effects captured by the checkpoint (weights) or recomputed per
    # in-window step (lr_override) must not
    expect = {CramIntervention: True, EarlyBurstControl: True,
              FreezeIntervention: True, ResetIntervention: False,
              LRSpikeIntervention: False, ShiftedLRWindow: False}
    for cls, flag in expect.items():
        assert getattr(cls, "replay_fire", False) is flag, cls.__name__
    print("replay_fire markers OK")


def test_cram_replay_reconstruction() -> None:
    """A killed-and-resumed cram run must reconstruct the same stream
    sequence: replay-firing the trigger over steps 1..step0 (what
    Trainer._resume does) must leave the fresh loader in the same state
    the original loader reached."""
    with tempfile.TemporaryDirectory() as td:
        base_bin = Path(td) / "base.bin"
        skill_bin = Path(td) / "skill.bin"
        np.full(20_000, 1, dtype=np.uint16).tofile(base_bin)
        np.full(20_000, 2, dtype=np.uint16).tofile(skill_bin)
        specs = [
            StreamSpec("base", base_bin, 0, 0.9),
            StreamSpec("arith", skill_bin, 0, 0.1),
        ]

        def fresh():
            loader = MixedLoader(specs, block_size=8, micro_batch_size=2,
                                 num_workers=0, seed=1)
            tr = SimpleNamespace(streams=specs, loader=loader,
                                 cfg=TrainConfig(run_name="cramtest"))
            iv = CramIntervention(at_step=51, end_step=101, stream="arith",
                                  boosted_density=0.5)
            return loader, tr, iv

        loader_a, tr_a, iv_a = fresh()
        names_a = []
        for step in range(1, 151):
            iv_a(step, tr_a)
            names_a.append(loader_a.next_batch(step)[2])

        # "kill" at step0=80 (mid-cram), replay 1..80, continue 81..150
        step0 = 80
        loader_b, tr_b, iv_b = fresh()
        names_b = []
        for step in range(1, step0 + 1):   # replay: refire + walk stream
            iv_b(step, tr_b)
            names_b.append(loader_b.next_batch(step)[2])
        for step in range(step0 + 1, 151):  # normal continuation
            iv_b(step, tr_b)
            names_b.append(loader_b.next_batch(step)[2])
        assert names_a == names_b, "replayed stream diverged"

        import gc
        del loader_a, loader_b, tr_a, tr_b
        gc.collect()
    print("cram replay reconstruction OK")


def test_fork_config() -> None:
    from dataclasses import asdict

    from run_intervention import fork_config

    src = TrainConfig(skill="arith", delay_onset=0.5,
                      skill_exposure_frac=0.2, seed=1)
    ckpt = {"train_config": asdict(src), "step": 2500}
    fc = fork_config(ckpt, "ivreset2501")
    assert fc.run_id == src.run_id + "_ivreset2501"
    assert fc.forked_from == f"{src.run_id}/step_0002500"
    assert isinstance(fc.betas, tuple)
    assert fc.seed == src.seed and fc.total_steps == src.total_steps
    # forks must reject already-intervened sources
    bad = asdict(src)
    bad["intervention"] = "reset_embedding"
    try:
        fork_config({"train_config": bad, "step": 2500}, "x")
        raise AssertionError("accepted an already-intervened source")
    except AssertionError as e:
        if "accepted" in str(e):
            raise
    # suffixed sources keep their suffix in the fork's namespace
    src2 = TrainConfig(skill="arith", delay_onset=0.5, run_suffix="poolv2")
    fc2 = fork_config({"train_config": asdict(src2), "step": 500}, "iv")
    assert fc2.run_id.endswith("_poolv2_iv")
    print("fork config OK")


if __name__ == "__main__":
    test_reset()
    test_lr_spike()
    test_freeze()
    test_cram()
    test_shifted_lr_window()
    test_replay_fire_markers()
    test_cram_replay_reconstruction()
    test_fork_config()
    print("ALL INTERVENTION TESTS PASSED")
