"""Read out an n=3 family of 23M d50 intervention arms.

Serves two pre-registered families:
  --family arms_n3    notes/intervention-arms-n3-preregistration.md   (T1.2)
  --family lr_decomp  notes/lr-decomposition-preregistration.md       (T1.4)

Reporting follows the project's standing rules, which are not optional here:

* **Absolute endpoints, never percent-of-gap.** The denominator would be the
  seed-variable null (per-seed range 0.52-0.63), so a recovery percentage moves
  even when the underlying result does not.
* **The null endpoint is printed BESIDE the treated endpoint, as two numbers**,
  never collapsed into one ratio — a collapsed figure hides which side moved.
* **Each arm is paired against its OWN seed's null** (the run it forked from),
  not against a pooled mean. That is what the common-checkpoint rule buys.
* **Perplexity travels with accuracy.** An arm that clears the accuracy gate
  while degrading held-out perplexity is forgetting, not reopening.

Endpoints come from each run's own logs/{run_id}/metrics.csv step-5000 row.
That column is scored on data/skills/arith/eval.jsonl — the single canonical
held-out draw — for every seed, so it is not affected by the per-seed eval
contamination that the Week-8 circuit tooling had.

Usage:  python scripts/arm_endpoints_n3.py --family arms_n3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import LOG_DIR  # noqa: E402

SEEDS = [1, 2, 3]
FINAL_STEP = 5000
NULL_TPL = "arith_delay50_exp20_size23M_seed{s}"
ANCHOR_TPL = "arith_delay0_exp20_size23M_seed{s}"
GATE_FRAC = 0.90          # reopening gate = 0.90 x early anchor
PPL_COST_GATE = 1.0       # forgetting gate, ppl above the arm's own null

FAMILIES = {
    "arms_n3": {
        "prereg": "notes/intervention-arms-n3-preregistration.md",
        "out": "logs/intervention_arms_n3/readout.json",
        "arms": [
            ("reset (embedding)", "_ivresetembedding2501"),
            ("reset + spike", "_ivresetspikeembedding2501"),
            ("LR spike", "_ivspike2501"),
            ("freeze", "_ivfreeze2501"),
            ("cram", "_ivcram2501"),
        ],
    },
    "lr_decomp": {
        "prereg": "notes/lr-decomposition-preregistration.md",
        "out": "logs/lr_decomposition/readout.json",
        "arms": [
            ("C flat @ late level", "_ivlrflatcontrol2501"),
            ("A level only", "_ivlrlevelonly2501"),
            ("B shape only", "_ivlrshapeonly2501"),
            ("lr_shift (both)", "_ivlrshift0at2501"),   # reference, already n=3
        ],
    },
}


def final_row(run_id: str) -> dict | None:
    f = LOG_DIR / run_id / "metrics.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df.step == FINAL_STEP]
    if df.empty:
        return None
    r = df.iloc[-1]
    return {"acc": float(r.arith_accuracy), "ppl": float(r.val_ppl)}


def desc(x: list[float]) -> dict:
    a = np.asarray(x, float)
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
            "n": int(a.size), "values": [float(v) for v in a]}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, (m - i) * p)
        adj[k] = min(1.0, run)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    args = ap.parse_args()
    fam = FAMILIES[args.family]

    nulls = {s: final_row(NULL_TPL.format(s=s)) for s in SEEDS}
    anchors = {s: final_row(ANCHOR_TPL.format(s=s)) for s in SEEDS}
    missing_base = [s for s in SEEDS if not nulls[s] or not anchors[s]]
    if missing_base:
        sys.exit(f"ABORT: missing null/anchor runs for seeds {missing_base}")

    null_acc = [nulls[s]["acc"] for s in SEEDS]
    anchor_acc = [anchors[s]["acc"] for s in SEEDS]
    gate = GATE_FRAC * float(np.mean(anchor_acc))

    print("=" * 78)
    print(f"n=3 ARM READOUT - family '{args.family}'")
    print(f"  {fam['prereg']}")
    print("=" * 78)
    print("\nNULL IDENTITY: each arm is paired against the run it FORKED FROM\n"
          "  (arith_delay50_exp20_size23M_seed{S}) - the only null a\n"
          "  common-checkpoint comparison may use. At seed 1 that is the 0.506\n"
          "  original, NOT the 0.524 `_recheck` re-train that serves as the\n"
          "  canonical Detect null elsewhere. Same run_id -> same model;\n"
          "  different run_id -> different model. The null printed here\n"
          "  (0.571 +/- 0.061) is the fork-source null and is EXPECTED to\n"
          "  differ from the Detect table's 0.577 +/- 0.051. Do not reconcile.\n")
    print(f"early anchor  {desc(anchor_acc)['mean']:.3f} "
          f"+/- {desc(anchor_acc)['sd']:.3f}   (per seed: "
          + ", ".join(f"{v:.3f}" for v in anchor_acc) + ")")
    print(f"delayed null  {desc(null_acc)['mean']:.3f} "
          f"+/- {desc(null_acc)['sd']:.3f}   (per seed: "
          + ", ".join(f"{v:.3f}" for v in null_acc) + ")")
    print(f"null cross-seed range [{min(null_acc):.3f}, {max(null_acc):.3f}]")
    print(f"reopening gate = {GATE_FRAC:g} x anchor = {gate:.3f}")

    rows, pvals, pending = {}, {}, []
    for label, suffix in fam["arms"]:
        got = {s: final_row(NULL_TPL.format(s=s) + suffix) for s in SEEDS}
        have = [s for s in SEEDS if got[s]]
        if not have:
            pending.append(label)
            continue
        acc = [got[s]["acc"] for s in have]
        lift = [got[s]["acc"] - nulls[s]["acc"] for s in have]
        ppl_cost = [got[s]["ppl"] - nulls[s]["ppl"] for s in have]
        d = desc(acc)
        entry = {
            "n": len(have), "seeds": have,
            "treated_endpoint": d,
            "null_endpoint": desc([nulls[s]["acc"] for s in have]),
            "per_seed_lift_over_own_null": {str(s): float(v)
                                            for s, v in zip(have, lift)},
            "lift_mean": float(np.mean(lift)),
            "ppl_cost_vs_own_null": {str(s): float(v)
                                     for s, v in zip(have, ppl_cost)},
            "ppl_cost_mean": float(np.mean(ppl_cost)),
            "clears_gate": bool(d["mean"] >= gate),
            "below_entire_null_range": bool(max(acc) < min(null_acc)),
            "all_seeds_negative_lift": bool(all(v < 0 for v in lift)),
            "sign_consistent": bool(len(set(np.sign(lift))) == 1),
        }
        if len(have) >= 2:
            t, p = stats.ttest_rel(acc, [nulls[s]["acc"] for s in have])
            entry["paired_t"] = float(t)
            entry["p_two_sided"] = float(p)
            pvals[label] = float(p)
        rows[label] = entry

    if pvals:
        adj = holm(pvals)
        for k, v in adj.items():
            rows[k]["p_holm"] = v

    print(f"\n{'arm':<22}{'n':>2} {'endpoint':>16} {'null':>16} "
          f"{'lift':>8} {'gate':>6} {'d_ppl':>8}")
    print("-" * 78)
    for label, e in rows.items():
        d, nu = e["treated_endpoint"], e["null_endpoint"]
        sd = "" if np.isnan(d["sd"]) else f" +/- {d['sd']:.3f}"
        nsd = "" if np.isnan(nu["sd"]) else f" +/- {nu['sd']:.3f}"
        print(f"{label:<22}{e['n']:>2} {d['mean']:>8.3f}{sd:<8} "
              f"{nu['mean']:>7.3f}{nsd:<8} {e['lift_mean']:>+7.3f} "
              f"{'PASS' if e['clears_gate'] else 'fail':>6} "
              f"{e['ppl_cost_mean']:>+7.2f}")

    print("\nper-seed lift over that seed's OWN null (sign consistency is the "
          "readable part; a near-zero mean with disagreeing signs is an "
          "absence of effect, not a small effect):")
    for label, e in rows.items():
        per = e["per_seed_lift_over_own_null"]
        print(f"  {label:<22} " + "  ".join(f"s{k}:{v:+.3f}"
                                            for k, v in per.items())
              + ("   [signs agree]" if e["sign_consistent"]
                 else "   [signs DISAGREE -> no consistent direction]"))

    print("\nreadings under the pre-registered rules:")
    for label, e in rows.items():
        if e["n"] < 3:
            note = f"n={e['n']} - stays SUGGESTIVE, cannot promote"
        elif e["clears_gate"]:
            note = ("REOPENS (clears gate)"
                    + ("" if e["ppl_cost_mean"] <= PPL_COST_GATE
                       else f" but ppl cost {e['ppl_cost_mean']:+.2f} > "
                            f"{PPL_COST_GATE} -> score as FORGETTING"))
        elif e["below_entire_null_range"] and e["all_seeds_negative_lift"]:
            note = "HARMFUL - below the entire null range at every seed"
        elif not e["sign_consistent"]:
            note = ("no consistent direction at n=3 - stays SUGGESTIVE, "
                    "do NOT write 'fails to reopen'")
        else:
            note = "does not reach the reopening gate"
        p = e.get("p_holm")
        print(f"  {label:<22} {note}" + (f"   (Holm p={p:.4f})" if p else ""))

    if pending:
        print(f"\nPENDING (no step-{FINAL_STEP} row yet): "
              + ", ".join(pending))

    out = Path(fam["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "family": args.family, "prereg": fam["prereg"],
        "gate": gate, "gate_frac": GATE_FRAC,
        "early_anchor": desc(anchor_acc), "delayed_null": desc(null_acc),
        "null_cross_seed_range": [min(null_acc), max(null_acc)],
        "arms": rows, "pending": pending,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
