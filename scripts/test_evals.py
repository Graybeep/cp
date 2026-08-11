"""Unit test for the eval harness (random model -> chance-level scores).

Run:  python scripts/test_evals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import ModelConfig  # noqa: E402
from reopen.evals import EvalHarness  # noqa: E402
from reopen.model import GPT  # noqa: E402


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    harness = EvalHarness(device=device, n_per_skill=64)
    assert harness.arith_problems, "arith eval set missing"
    assert harness.blimp, "blimp paradigms missing"
    assert harness.vocab_instances, "vocab eval set missing"

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=harness.tok.get_vocab_size(), n_layer=2,
                      n_head=8, n_embd=128, block_size=512)
    model = GPT(cfg).to(device)

    res = harness.evaluate(model)
    assert 0.0 <= res["arith_accuracy"] <= 0.05
    # The licensing interaction must sit at chance for a model with no
    # syntax — that is the whole point of the bias correction. A raw
    # wh_vs_that paradigm would instead read ~0.0 or ~1.0 here purely
    # from the "that" frequency prior.
    for key in ("syntax_accuracy", "syntax_accuracy_local"):
        assert 0.4 <= res[key] <= 0.6, f"{key} should be ~chance: {res[key]}"
    assert 0.0 <= res["vocab_control_accuracy"] <= 0.1
    assert res["vocab_control_logprob"] < -5
    assert set(res["blimp"]) == set(harness.primaries)

    res_pw = harness.evaluate(model, per_word=True)
    assert "vocab_per_word" in res_pw and len(res_pw["vocab_per_word"]) > 0

    print(f"harness OK: arith {res['arith_accuracy']:.3f}, "
          f"syntax {res['syntax_accuracy']:.3f}, "
          f"vocab {res['vocab_control_accuracy']:.3f} "
          f"(lp {res['vocab_control_logprob']:.1f})")
    print("ALL EVAL HARNESS TESTS PASSED")


if __name__ == "__main__":
    main()
