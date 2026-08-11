"""Compare final-checkpoint metrics across runs.

Prints skill accuracy and general perplexity side by side — the pairing
that distinguishes real recovery/acquisition from a capability tradeoff.
Reads posthoc_evals.csv when present (full eval sets), else metrics.csv.

Run:  python scripts/compare_runs.py run_id_a run_id_b [...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import LOG_DIR  # noqa: E402

COLS = [
    ("syntax_accuracy", "syntax_LD", "{:.3f}"),
    ("syntax_accuracy_local", "syntax_local", "{:.3f}"),
    ("arith_accuracy", "arith", "{:.3f}"),
    ("vocab_control_accuracy", "vocab_acc", "{:.3f}"),
    ("vocab_control_logprob", "vocab_lp", "{:.2f}"),
    ("val_ppl", "val_ppl", "{:.1f}"),
]


def load_final(run_id: str) -> dict:
    d = LOG_DIR / run_id
    row: dict = {"run_id": run_id}
    posthoc = d / "posthoc_evals.csv"
    metrics = d / "metrics.csv"
    if posthoc.exists():
        df = pd.read_csv(posthoc).sort_values("step")
        row.update(df.iloc[-1].to_dict())
        row["source"] = "posthoc"
    if metrics.exists():
        df = pd.read_csv(metrics).sort_values("step")
        last = df.iloc[-1].to_dict()
        # metrics.csv carries val_ppl / skill_tokens; posthoc does not
        for k, v in last.items():
            row.setdefault(k, v)
        row["val_ppl"] = last.get("val_ppl")
        row["skill_tokens_seen"] = last.get("skill_tokens_seen")
        row["step"] = last.get("step", row.get("step"))
        row.setdefault("source", "metrics")
    if len(row) == 1:
        sys.exit(f"no metrics found for {run_id}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    args = ap.parse_args()

    rows = [load_final(r) for r in args.run_ids]
    width = max(len(r["run_id"]) for r in rows) + 2

    header = f"{'run':<{width}}{'step':>6}{'skill_tok':>11}"
    for _key, label, _fmt in COLS:
        header += f"{label:>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        st = r.get("skill_tokens_seen")
        line = (f"{r['run_id']:<{width}}{int(r['step']):>6}"
                f"{int(st) if pd.notna(st) else 0:>11,}")
        for key, _label, fmt in COLS:
            v = r.get(key)
            line += f"{(fmt.format(v) if pd.notna(v) else '-'):>14}"
        print(line)

    if len(rows) == 2:
        a, b = rows
        print("\ndelta (second - first):")
        for key, label, fmt in COLS:
            va, vb = a.get(key), b.get(key)
            if pd.notna(va) and pd.notna(vb):
                print(f"  {label:<16}{vb - va:+.3f}")


if __name__ == "__main__":
    main()
