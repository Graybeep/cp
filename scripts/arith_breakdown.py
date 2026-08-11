"""Diagnostic: per-difficulty / per-token breakdown of arith exact-match.

For each selected checkpoint of a run, grades the FULL arith eval set and
reports, per (digit-bucket x carry) cell:
  - exact-match accuracy (the metric logged in metrics.csv)
  - per-answer-token accuracy (graded signal exact-match hides)
  - mean answer-token log-prob (fully continuous signal)

Usage:
  python arith_breakdown.py <run_id> [--every 500] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(r"D:\EXPERIMENTS\reopen")
sys.path.insert(0, str(PROJECT_ROOT))

from reopen.checkpoint import load_checkpoint  # noqa: E402
from reopen.config import CHECKPOINT_DIR, DATA_DIR, LOG_DIR, ModelConfig  # noqa: E402
from reopen.incremental import IncrementalCSV  # noqa: E402
from reopen.model import GPT  # noqa: E402
from reopen.skills.arithmetic import _answer_span  # noqa: E402


def load_eval_problems() -> list[dict]:
    path = DATA_DIR / "skills" / "arith" / "eval.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def digit_bucket(a: int, b: int) -> int:
    return max(len(str(a)), len(str(b)))


@torch.no_grad()
def grade_detailed(model, tok, problems: list[dict], device: str,
                   batch_size: int = 64, prompt_prefix: str = "\n") -> pd.DataFrame:
    """Per-problem: exact match, n answer tokens correct, mean logprob."""
    model.eval()
    recs = []
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i + batch_size]
        rows, spans = [], []
        for p in batch:
            prompt = f"{prompt_prefix}{p['a']}+{p['b']}="
            ids, ans_start = _answer_span(tok, prompt, prompt + str(p['a'] + p['b']))
            rows.append(ids)
            spans.append((ans_start, len(ids) - ans_start))
        maxlen = max(len(r) for r in rows)
        x = torch.zeros(len(rows), maxlen, dtype=torch.long)
        for r, ids in enumerate(rows):
            x[r, :len(ids)] = torch.tensor(ids)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x.to(device))
        logprobs = F.log_softmax(logits.float(), dim=-1).cpu()
        preds = logits.argmax(-1).cpu()
        for r, (ans_start, n_ans) in enumerate(spans):
            tok_ok, lp_sum = 0, 0.0
            for j in range(n_ans):
                target = rows[r][ans_start + j]
                pos = ans_start - 1 + j
                tok_ok += int(preds[r, pos] == target)
                lp_sum += logprobs[r, pos, target].item()
            p = batch[r]
            recs.append({
                "digits": digit_bucket(p["a"], p["b"]),
                "carry": p["carry"],
                "exact": int(tok_ok == n_ans),
                "tok_correct": tok_ok,
                "n_tok": n_ans,
                "mean_logprob": lp_sum / n_ans,
            })
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    problems = load_eval_problems()
    print(f"{len(problems)} eval problems")

    ckpt_dir = CHECKPOINT_DIR / args.run_id
    paths = [p for p in sorted(ckpt_dir.glob("step_*.pt"))
             if int(p.stem.split("_")[1]) % args.every == 0]
    assert paths, f"no checkpoints under {ckpt_dir} at interval {args.every}"

    out_dir = LOG_DIR / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = IncrementalCSV(out_dir / "arith_breakdown.csv", key="step")
    if out_csv.done:
        print(f"resuming: {len(out_csv.done)} checkpoints already graded")

    model = None
    for path in paths:
        if int(path.stem.split("_")[1]) in out_csv.done:
            continue
        ckpt = load_checkpoint(path, device="cpu")
        if model is None:
            model = GPT(ModelConfig(**ckpt["model_config"])).to(args.device)
        model.load_state_dict(ckpt["model"])
        df = grade_detailed(model, tok, problems, args.device)
        df["step"] = ckpt["step"]
        out_csv.append(df.to_dict("records"))
        agg = df.agg({"exact": "mean", "tok_correct": "sum",
                      "n_tok": "sum", "mean_logprob": "mean"})
        print(f"step {ckpt['step']:5d}: exact={agg['exact']:.3f} "
              f"tok_acc={agg['tok_correct'] / agg['n_tok']:.3f} "
              f"logprob={agg['mean_logprob']:.3f}")

    full = out_csv.frame()

    final = full[full.step == full.step.max()]
    cell = final.groupby(["digits", "carry"]).apply(
        lambda g: pd.Series({
            "n": len(g),
            "exact": g.exact.mean(),
            "tok_acc": g.tok_correct.sum() / g.n_tok.sum(),
            "logprob": g.mean_logprob.mean(),
        }), include_groups=False)
    print(f"\n=== final checkpoint (step {full.step.max()}) by difficulty ===")
    print(cell.to_string(float_format=lambda v: f"{v:.3f}"))

    traj = full.groupby(["step", "digits", "carry"]).apply(
        lambda g: g.tok_correct.sum() / g.n_tok.sum(), include_groups=False
    ).unstack(["digits", "carry"])
    print("\n=== per-token accuracy trajectory by (digits, carry) ===")
    print(traj.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_csv.path} ({len(out_csv.done)} checkpoints)")


if __name__ == "__main__":
    main()
