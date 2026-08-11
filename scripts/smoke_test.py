"""Fast smoke test (no data needed): tiny model forward/backward, RoPE
shapes, per-component resets, checkpoint save/load roundtrip, LR schedule.

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from reopen.config import ModelConfig, TrainConfig  # noqa: E402
from reopen.model import GPT  # noqa: E402
from reopen.train import lr_at_step  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=512, n_layer=2, n_head=8, n_embd=128,
                      block_size=64)
    model = GPT(cfg)
    print(f"params: {model.num_params():,}")

    # untied check
    assert model.wte.weight.data_ptr() != model.lm_head.weight.data_ptr(), \
        "embedding and lm_head must be UNTIED"

    x = torch.randint(0, cfg.vocab_size, (4, 64))
    y = torch.randint(0, cfg.vocab_size, (4, 64))
    logits, loss = model(x, y)
    assert logits.shape == (4, 64, cfg.vocab_size)
    loss.backward()
    print(f"forward/backward OK, loss {loss.item():.3f} "
          f"(expect ~ln(512)={torch.log(torch.tensor(512.)).item():.3f})")

    # per-component reset only changes the targeted component
    head_before = model.lm_head.weight.clone()
    emb_before = model.wte.weight.clone()
    model.reset_embedding()
    assert not torch.equal(model.wte.weight, emb_before), "wte not reset"
    assert torch.equal(model.lm_head.weight, head_before), \
        "reset_embedding must not touch lm_head"
    block0_before = [p.clone() for p in model.blocks[0].parameters()]
    block1_before = [p.clone() for p in model.blocks[1].parameters()]
    model.reset_block(0)
    assert any(not torch.equal(a, b) for a, b in
               zip(model.blocks[0].parameters(), block0_before))
    assert all(torch.equal(a, b) for a, b in
               zip(model.blocks[1].parameters(), block1_before)), \
        "reset_block(0) must not touch block 1"
    print("per-component resets OK (isolated)")

    # checkpoint roundtrip with metadata
    tcfg = TrainConfig(run_name="smoke", total_steps=100,
                       checkpoint_interval=50, warmup_steps=10)
    opt = model.configure_optimizer(tcfg.lr, tcfg.weight_decay, tcfg.betas,
                                    device="cpu")
    with tempfile.TemporaryDirectory() as td:
        p = save_checkpoint(model, opt, step=42, model_cfg=cfg,
                            train_cfg=tcfg, base_dir=Path(td))
        assert p.name == "step_0000042.pt", p.name
        ckpt = load_checkpoint(p)
        assert ckpt["step"] == 42
        assert ckpt["metadata"]["intervention"] is None
        m2 = GPT(cfg)
        m2.load_state_dict(ckpt["model"])
        assert torch.equal(m2.lm_head.weight, model.lm_head.weight)
    print("checkpoint roundtrip OK (naming + metadata + weights)")

    # LR schedule shape
    lrs = [lr_at_step(s, tcfg) for s in range(tcfg.total_steps)]
    peak = max(lrs)
    assert abs(peak - tcfg.lr) / tcfg.lr < 0.05, "peak LR should be ~cfg.lr"
    assert lrs[-1] < lrs[len(lrs) // 2] < peak, "LR must decay after warmup"
    assert lrs[-1] >= tcfg.lr * tcfg.min_lr_ratio * 0.99, "cosine floor"
    print("LR schedule OK (warmup -> cosine -> floor)")

    if torch.cuda.is_available():
        model = model.cuda()
        x, y = x.cuda(), y.cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        print(f"CUDA bf16 forward OK, loss {loss.item():.3f}")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
