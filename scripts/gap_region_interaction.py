"""Continuation-scored wh-licensing interaction (case-3(b) metric audit).

The standard metric compares whole-sentence log-probs of minimal pairs
that differ only at the complementizer ("what" vs "that"). That makes the
score sensitive to the model's complementizer PRIOR in embedded position —
demonstrated empirically by the pool-v2 probe, where flooding training
with embed-verb+wh sequences drove the interaction to ~0.5 without any
change to gap knowledge.

Fix: both members of a pair are IDENTICAL after the complementizer, so
score only the shared continuation, conditioned on each prefix:

    with_gap: P(continuation | ...wh...)  >  P(continuation | ...that...)
    no_gap:   P(continuation | ...that...) >  P(continuation | ...wh...)

The complementizer's own probability is in the condition, not the score,
so its unigram/contextual prior cancels exactly; what remains is whether
the model treats a downstream gap as contingent on the filler — the
licensing dependency itself (gap-region surprisal, Wilcox-style).

Run:  python scripts/gap_region_interaction.py <run_id> [<run_id> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.checkpoint import load_checkpoint  # noqa: E402
from reopen.config import CHECKPOINT_DIR, DATA_DIR, ModelConfig  # noqa: E402
from reopen.incremental import atomic_write_text, load_json_valid  # noqa: E402
from reopen.model import GPT  # noqa: E402
from reopen.skills import syntax  # noqa: E402


@torch.no_grad()
def continuation_logprobs(model, tok, pairs, device: str,
                          batch_size: int = 64) -> list[tuple[float, float]]:
    """Per pair: (logP of shared continuation | good prefix,
                  logP of shared continuation | bad prefix)."""
    rows, spans = [], []
    for good, bad in pairs:
        g, b = tok.encode(good).ids, tok.encode(bad).ids
        # shared suffix length (the continuation after the complementizer)
        k = 0
        while (k < min(len(g), len(b)) - 1 and g[-1 - k] == b[-1 - k]):
            k += 1
        rows.append((g, b, k))
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        seqs = [s for g, b, _k in batch for s in (g, b)]
        maxlen = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        for r, ids in enumerate(seqs):
            x[r, : len(ids)] = torch.tensor(ids)
        logits, _ = model(x.to(device))
        logp = torch.log_softmax(logits.float(), dim=-1).cpu()
        for j, (g, b, k) in enumerate(batch):
            def cont_lp(ids, row):
                # tokens ids[-k:] scored from positions len(ids)-k-1 ...
                s = 0.0
                for t in range(len(ids) - k, len(ids)):
                    s += logp[row, t - 1, ids[t]].item()
                return s
            out.append((cont_lp(g, 2 * j), cont_lp(b, 2 * j + 1)))
    return out


def evaluate(run_id: str, tok, device: str, every: int = 0,
             model_cache: dict = {}, save_items: Path | None = None) -> None:
    paths = sorted((CHECKPOINT_DIR / run_id).glob("step_*.pt"))
    if every:
        paths = [p for p in paths
                 if int(p.stem.split("_")[1]) % every == 0]
    else:
        paths = paths[-1:]
    for path in paths:
        if save_items is not None:
            step = int(path.stem.split("_")[1])
            existing = save_items / f"{run_id}_step{step:07d}_items.json"
            # skip only if the file exists AND parses — a kill mid-write
            # must not leave a truncated file that looks "done"
            if load_json_valid(existing) is not None:
                continue
        _evaluate_ckpt(run_id, path, tok, device, model_cache,
                       compact=bool(every), save_items=save_items)


def _evaluate_ckpt(run_id: str, path, tok, device: str,
                   model_cache: dict, compact: bool,
                   save_items: Path | None = None) -> None:
    ckpt = load_checkpoint(path, device="cpu")
    if "model" not in model_cache:
        model_cache["model"] = GPT(
            ModelConfig(**ckpt["model_config"])).to(device)
    model = model_cache["model"]
    model.load_state_dict(ckpt["model"])
    model.eval()

    accs = {}
    items = {}
    for name in ["wh_vs_that_with_gap", "wh_vs_that_no_gap",
                 "wh_vs_that_with_gap_long_distance",
                 "wh_vs_that_no_gap_long_distance"]:
        pairs = syntax.load_blimp_pairs(DATA_DIR / "blimp" / f"{name}.jsonl")
        lps = continuation_logprobs(model, tok, pairs, device)
        # good member has wh in with_gap paradigms and 'that' in no_gap
        # paradigms, so "good continuation more probable" is the correct
        # direction in both
        wins = [int(g > b) for g, b in lps]
        items[name] = wins
        accs[name] = sum(wins) / len(wins)
    if save_items is not None:
        save_items.mkdir(parents=True, exist_ok=True)
        out = save_items / f"{run_id}_step{ckpt['step']:07d}_items.json"
        atomic_write_text(out, json.dumps(
            {"run_id": run_id, "step": ckpt["step"], "items": items}))

    local = (accs["wh_vs_that_with_gap"] + accs["wh_vs_that_no_gap"]) / 2
    ld = (accs["wh_vs_that_with_gap_long_distance"]
          + accs["wh_vs_that_no_gap_long_distance"]) / 2
    if compact:
        print(f"{run_id} step {ckpt['step']:6d}: "
              f"local={local:.3f}  long-distance={ld:.3f}")
        return
    print(f"{run_id} @ step {ckpt['step']}")
    for k, v in accs.items():
        print(f"  {k:45s} {v:.3f}")
    print(f"  continuation interaction: local={local:.3f}  "
          f"long-distance={ld:.3f}   (chance/prior-only = 0.5)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--every", type=int, default=0,
                    help="score every Nth-step checkpoint (0 = final only)")
    ap.add_argument("--save-items", default=None,
                    help="directory to write per-item win arrays "
                         "(enables paired McNemar tests across runs)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    for rid in args.run_ids:
        evaluate(rid, tok, args.device, every=args.every, model_cache={},
                 save_items=Path(args.save_items) if args.save_items else None)


if __name__ == "__main__":
    main()
