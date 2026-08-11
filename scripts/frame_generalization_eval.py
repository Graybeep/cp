"""Structural-leak test for the recombined syntax pool.

Compares a checkpoint's per-token log-prob on two matched sets of NOVEL
sentences (neither trained on):
  eval_trained: recombinations of frames the model saw in training
  eval_heldout: recombinations of frames excluded from training entirely

trained >> heldout  -> the model memorized frame skeletons; a dose-driven
                       LD improvement is suspect (leak one level below
                       lexical dedup) -> Tier 2 synthetic generator.
small gap           -> the construction generalized; dose results stand.

Run:  python scripts/frame_generalization_eval.py <run_id> [--step N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.checkpoint import load_checkpoint  # noqa: E402
from reopen.config import CHECKPOINT_DIR, DATA_DIR, ModelConfig  # noqa: E402
from reopen.model import GPT  # noqa: E402
from reopen.skills.syntax import _sentence_logprobs  # noqa: E402


def per_token_logprobs(model, tok, sentences: list[str], device: str) -> np.ndarray:
    totals = _sentence_logprobs(model, tok, sentences, device, batch_size=64)
    n_tok = np.array([len(tok.encode(s).ids) - 1 for s in sentences])
    return totals.numpy() / np.maximum(n_tok, 1)


def summarize(name: str, recs: list[dict], vals: np.ndarray) -> None:
    ld = np.array([r["long_distance"] for r in recs], dtype=bool)
    print(f"{name}: mean={vals.mean():.4f}  (n={len(vals)}, "
          f"local={vals[~ld].mean():.4f} n={int((~ld).sum())}, "
          f"long-distance={vals[ld].mean():.4f} n={int(ld.sum())})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: latest)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ev = json.loads((DATA_DIR / "skills" / "syntax" /
                     "frame_generalization_eval.json").read_text())
    # two formats: flat (matrix only, pre-embedded pools) or per-level
    levels = ({"matrix": ev} if "eval_trained" in ev else
              {k: v for k, v in ev.items() if isinstance(v, dict)
               and "eval_trained" in v})

    ckpt_dir = CHECKPOINT_DIR / args.run_id
    if args.step is None:
        path = sorted(ckpt_dir.glob("step_*.pt"))[-1]
    else:
        path = ckpt_dir / f"step_{args.step:07d}.pt"
    ckpt = load_checkpoint(path, device="cpu")
    model = GPT(ModelConfig(**ckpt["model_config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))

    # Licensing contrast: does the grammaticality preference itself
    # generalize to unseen frames? Forced choice, each sentence vs its
    # wh->'that' corruption — the FIRST what/who/whom is replaced, which
    # handles both matrix ('what did you see ?' -> 'that did you see ?')
    # and embedded ('i wonder what thomas has .' -> 'i wonder that thomas
    # has .'; the embedded case is structurally the BLiMP wh_vs_that
    # contrast). The corruption and any frequency prior on 'that' are
    # identical for both sets, so the TRAINED-vs-HELDOUT accuracy
    # difference isolates frame-boundness of the licensing knowledge.
    wh_re = __import__("re").compile(r"\b(what|who|whom)\b")

    def licensing_acc(recs: list[dict]) -> tuple[float, int, np.ndarray]:
        pairs = [(r["sentence"], wh_re.sub("that", r["sentence"], count=1))
                 for r in recs if wh_re.search(r["sentence"])]
        good = _sentence_logprobs(model, tok, [g for g, _b in pairs],
                                  args.device, 64).numpy()
        bad = _sentence_logprobs(model, tok, [b for _g, b in pairs],
                                 args.device, 64).numpy()
        wins = (good > bad)
        return wins.mean(), len(pairs), wins

    for level, data in levels.items():
        trained, heldout = data["eval_trained"], data["eval_heldout"]
        print(f"\n===== level: {level}  (trained-frame n={len(trained)}, "
              f"heldout-frame n={len(heldout)})")
        vt = per_token_logprobs(model, tok, [r["sentence"] for r in trained],
                                args.device)
        vh = per_token_logprobs(model, tok, [r["sentence"] for r in heldout],
                                args.device)
        summarize("trained-frame (novel sentences)", trained, vt)
        summarize("heldout-frame (novel sentences)", heldout, vh)

        gap = vt.mean() - vh.mean()
        u = sps.mannwhitneyu(vt, vh, alternative="two-sided")
        pooled = np.sqrt((vt.var(ddof=1) + vh.var(ddof=1)) / 2)
        d = gap / pooled if pooled > 0 else float("nan")
        print(f"raw-fit gap (trained - heldout): {gap:+.4f} nats/token  "
              f"cohen_d={d:.3f}  mann-whitney p={u.pvalue:.2e}")
        print("(raw fit = string familiarity; the licensing contrast is "
              "the decision criterion)")

        at, nt, wt = licensing_acc(trained)
        ah, nh, wh = licensing_acc(heldout)
        print("licensing preference (good > wh->that corruption):")
        print(f"  trained-frame: {at:.3f} (n={nt})")
        print(f"  heldout-frame: {ah:.3f} (n={nh})")
        ct = sps.chi2_contingency([[wt.sum(), len(wt) - wt.sum()],
                                   [wh.sum(), len(wh) - wh.sum()]])
        print(f"  difference: {at - ah:+.3f}  chi2 p={ct.pvalue:.3g}")
    print("\nread: heldout ~= trained -> licensing generalized (dose result "
          "stands); trained >> heldout -> licensing is frame-bound "
          "memorization -> Tier 2 synthetic.")


if __name__ == "__main__":
    main()
