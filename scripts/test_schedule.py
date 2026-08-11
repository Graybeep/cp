"""Unit tests for the delay scheduler (plain asserts).

Run:  python scripts/test_schedule.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import ModelConfig, TrainConfig  # noqa: E402
from reopen.schedule import (  # noqa: E402
    MixedLoader, StreamSpec, exposure_plan, resolve_streams,
)


def test_resolve_and_matched_exposure() -> None:
    mcfg = ModelConfig()

    base = TrainConfig(skill="none")
    specs = resolve_streams(base)
    assert len(specs) == 1 and specs[0].name == "base"

    plans = {}
    for d in (0.0, 0.25, 0.5, 0.75):
        cfg = TrainConfig(skill="arith", delay_onset=d, total_steps=4000,
                          skill_exposure_frac=0.05)
        specs = resolve_streams(cfg)
        assert specs[1].onset_step == int(round(d * 4000))
        assert abs(specs[1].density - 0.05 / (1 - d)) < 1e-12
        plans[d] = exposure_plan(specs, cfg, mcfg)

    tokens = {d: p["streams"]["arith"]["planned_tokens"]
              for d, p in plans.items()}
    assert len(set(tokens.values())) == 1, (
        f"total skill exposure must be matched across delays: {tokens}"
    )
    base_tokens = {d: p["streams"]["base"]["planned_tokens"]
                   for d, p in plans.items()}
    assert len(set(base_tokens.values())) == 1, (
        f"base exposure must be matched across delays too: {base_tokens}"
    )

    # exposure so high that a late onset would dominate the stream -> error
    try:
        resolve_streams(TrainConfig(skill="arith", delay_onset=0.9,
                                    skill_exposure_frac=0.1))
        raise SystemExit("expected density assertion to fire")
    except AssertionError:
        pass
    print("resolve/matched-exposure OK")


def _make_bin(path: Path, fill: int, n: int = 20_000) -> Path:
    np.full(n, fill, dtype=np.uint16).tofile(path)
    return path


def test_mixed_loader(density: float = 0.2) -> None:
    total_steps, accum, onset = 200, 2, 100
    with tempfile.TemporaryDirectory() as td:
        base_bin = _make_bin(Path(td) / "base.bin", fill=1)
        skill_bin = _make_bin(Path(td) / "skill.bin", fill=2)
        specs = [
            StreamSpec("base", base_bin, 0, 1 - density),
            StreamSpec("arith", skill_bin, onset, density),
        ]

        def run() -> list[str]:
            loader = MixedLoader(specs, block_size=8, micro_batch_size=2,
                                 num_workers=0, seed=1)
            seq = []
            for step in range(1, total_steps + 1):
                for _ in range(accum):
                    x, y, name = loader.next_batch(step)
                    seq.append(name)
                    fill = 2 if name == "arith" else 1
                    assert (x == fill).all() and (y == fill).all(), (
                        f"stream {name} yielded wrong data"
                    )
            return seq

        seq = run()
        pre = seq[: onset * accum]
        post = seq[onset * accum:]
        assert all(n == "base" for n in pre), "skill leaked before onset"
        n_skill = sum(n == "arith" for n in post)
        expected = int((total_steps - onset) * accum * density)
        assert n_skill == expected, (
            f"error diffusion must realize the plan exactly: "
            f"{n_skill} != {expected}"
        )
        # evenly spaced: every consecutive window of 1/density micro-batches
        # after onset contains exactly one skill batch
        if (1 / density).is_integer():
            stride = int(1 / density)
            for i in range(0, len(post) - stride + 1, stride):
                assert sum(n == "arith" for n in post[i : i + stride]) == 1

        assert run() == seq, "schedule must be deterministic"
    print(f"mixed loader OK at density {density} "
          "(exact counts, spacing, deterministic)")


if __name__ == "__main__":
    test_resolve_and_matched_exposure()
    # 0.1 is the float-drift regression case: ten additions of 0.1 sum to
    # 0.9999...; the multiplicative bookkeeping must still emit exactly.
    for dens in (0.2, 0.1, 1 / 3, 0.05):
        test_mixed_loader(dens)
    print("ALL SCHEDULE TESTS PASSED")
