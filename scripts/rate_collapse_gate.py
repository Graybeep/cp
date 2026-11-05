"""60M syntax rate-collapse test — pre-registered gate.

Executes notes/rate-collapse-60M-preregistration.md verbatim on the 24 delayed
60M runs, against the CACHED 24-seed delay-0 baseline from the size sweep.

Primary   : one-sided Fisher exact on P(high) at the derived threshold 0.5593.
Secondary : Mann-Whitney U on the continuous LD distributions (threshold-free),
            + difference in means with a bootstrap CI.
Holm-corrected within that 2-test family. The 0.55 threshold is a robustness
check on the primary, reported uncorrected and labelled as such.
The 0.57 threshold is EXCLUDED A PRIORI as unpowered (best case p = 0.2447).

The null outcome is deliberately given a narrow reading: it excludes a FULL
collapse (power 1.000) and not a HALF collapse (power 0.188). The script prints
that reading rather than leaving it to the write-up.

Outputs a human report to stdout and a machine record to
logs/rate_collapse/gate_result.json (analysis-outputs-in-logs rule).

Run:  python scripts/rate_collapse_gate.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import CHECKPOINT_DIR, DATA_DIR  # noqa: E402
from scripts.gap_region_interaction import evaluate  # noqa: E402

SEEDS = list(range(1, 25))
PRIMARY_THR = 0.5593          # mu_low + 3*sigma_low, derived from the 11M low cluster
SECONDARY_THR = 0.55          # permissive robustness check
EXCLUDED_THR = 0.57           # unpowered by construction — reported, never tested
GUARD_FLOOR = 0.15            # neither LD paradigm may collapse to the prior

ITEMS_DIR = Path("logs/rate_collapse/items")
OUT_JSON = Path("logs/rate_collapse/gate_result.json")
BASELINE_CACHE = Path("logs/szsweep_eval/per_seed_ld.json")
BASELINE_ITEMS = Path("logs/szsweep_eval/items")

LD_WITH = "wh_vs_that_with_gap_long_distance"
LD_NO = "wh_vs_that_no_gap_long_distance"


def delayed_run_id(seed: int) -> str:
    return f"syntax_delay50_exp10_size60M_seed{seed}_ratecollapse"


def baseline_run_id(seed: int) -> str:
    return f"syntax_delay0_exp10_size60M_seed{seed}_szsweep"


def load_row(rid: str, items_dir: Path, tok, device: str) -> dict:
    """Per-run LD interaction + its two components, caching item arrays."""
    item_file = items_dir / f"{rid}_step0005000_items.json"
    if not item_file.exists():
        evaluate(rid, tok, device, every=0, model_cache={}, save_items=items_dir)
    items = json.loads(item_file.read_text())["items"]
    with_gap = float(np.mean(items[LD_WITH]))
    no_gap = float(np.mean(items[LD_NO]))
    return {"run_id": rid, "ld": (with_gap + no_gap) / 2,
            "ld_with_gap": with_gap, "ld_no_gap": no_gap}


def guarded_high(row: dict, thr: float) -> bool:
    return (row["ld"] >= thr
            and row["ld_with_gap"] >= GUARD_FLOOR
            and row["ld_no_gap"] >= GUARD_FLOOR)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, order-preserving."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-partial", action="store_true",
                    help="run at the achieved n if some delayed seeds are "
                         "missing (pre-reg §6: report realised n, never swap "
                         "seeds)")
    args = ap.parse_args()

    missing = [delayed_run_id(s) for s in SEEDS
               if not (CHECKPOINT_DIR / delayed_run_id(s)
                       / "step_0005000.pt").exists()]
    if missing and not args.allow_partial:
        sys.exit(f"ABORT: {len(missing)} of 24 delayed runs incomplete "
                 f"(e.g. {missing[:3]}). Pre-registration forbids peeking at "
                 f"the decision statistic before the arm is complete; pass "
                 f"--allow-partial only to report a genuinely truncated run.")

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)

    done = [s for s in SEEDS if (CHECKPOINT_DIR / delayed_run_id(s)
                                 / "step_0005000.pt").exists()]
    delayed = [load_row(delayed_run_id(s), ITEMS_DIR, tok, args.device)
               for s in done]

    # Baseline: reuse the cached size-sweep arrays. per_seed_ld.json holds the
    # LD means only, so recompute the guarded components from the cached item
    # arrays (already on disk from the size-sweep gate — no GPU work).
    base_rows = [load_row(baseline_run_id(s), BASELINE_ITEMS, tok, args.device)
                 for s in SEEDS]
    cached = json.loads(BASELINE_CACHE.read_text())["60M"]
    recomputed = sorted(round(r["ld"], 6) for r in base_rows)
    if not np.allclose(sorted(cached), recomputed, atol=1e-6):
        print("WARNING: recomputed baseline LD != cached per_seed_ld.json; "
              "using the recomputed values and flagging the mismatch.")

    d_ld = np.array([r["ld"] for r in delayed])
    b_ld = np.array([r["ld"] for r in base_rows])

    print("=" * 74)
    print("60M SYNTAX RATE-COLLAPSE TEST — PRE-REGISTERED GATE")
    print("  notes/rate-collapse-60M-preregistration.md")
    print("=" * 74)
    if missing:
        print(f"\n!! PARTIAL: {len(missing)} delayed runs missing; "
              f"realised n = {len(delayed)} (pre-reg §6)")
    print(f"\ndelayed  n={len(d_ld):>2}  mean={d_ld.mean():.4f}  "
          f"sd={d_ld.std(ddof=1):.4f}  min={d_ld.min():.3f}  max={d_ld.max():.3f}")
    print(f"baseline n={len(b_ld):>2}  mean={b_ld.mean():.4f}  "
          f"sd={b_ld.std(ddof=1):.4f}  min={b_ld.min():.3f}  max={b_ld.max():.3f}")

    # ---- PRIMARY: Fisher exact on the exceedance rate ---------------------
    results = {}
    for thr, tag in [(PRIMARY_THR, "PRIMARY"), (SECONDARY_THR, "robustness")]:
        dh = sum(guarded_high(r, thr) for r in delayed)
        bh = sum(guarded_high(r, thr) for r in base_rows)
        table = [[dh, len(delayed) - dh], [bh, len(base_rows) - bh]]
        p = stats.fisher_exact(table, alternative="less")[1]
        results[thr] = {"delayed_high": dh, "delayed_n": len(delayed),
                        "baseline_high": bh, "baseline_n": len(base_rows),
                        "p_one_sided": p, "tag": tag}
        print(f"\n[{tag}] threshold {thr}")
        print(f"  delayed  P(high) = {dh}/{len(delayed)} = {dh/len(delayed):.3f}")
        print(f"  baseline P(high) = {bh}/{len(base_rows)} = {bh/len(base_rows):.3f}")
        print(f"  one-sided Fisher exact p = {p:.4f}")

    # excluded threshold: reported for completeness, never tested
    dh_x = sum(guarded_high(r, EXCLUDED_THR) for r in delayed)
    bh_x = sum(guarded_high(r, EXCLUDED_THR) for r in base_rows)
    print(f"\n[EXCLUDED A PRIORI] threshold {EXCLUDED_THR}: "
          f"delayed {dh_x}/{len(delayed)}, baseline {bh_x}/{len(base_rows)} "
          f"— unpowered by construction (best case p = 0.2447), not tested")

    # ---- SECONDARY: threshold-free ---------------------------------------
    u, p_mw = stats.mannwhitneyu(d_ld, b_ld, alternative="less")
    diff = float(d_ld.mean() - b_ld.mean())
    rng = np.random.default_rng(20260811)
    boot = np.array([rng.choice(d_ld, len(d_ld), replace=True).mean()
                     - rng.choice(b_ld, len(b_ld), replace=True).mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n[SECONDARY, threshold-free] Mann-Whitney U={u:.1f} "
          f"one-sided p={p_mw:.4f}")
    print(f"  mean difference (delayed - baseline) = {diff:+.4f} "
          f"[95% CI {lo:+.4f}, {hi:+.4f}]")

    # ---- Holm within the declared family ---------------------------------
    fam = {"fisher_primary": results[PRIMARY_THR]["p_one_sided"],
           "mannwhitney": float(p_mw)}
    adj = holm(fam)
    print("\nHolm-corrected family (2 tests, pre-registered):")
    for k in fam:
        print(f"  {k:<16} raw p={fam[k]:.4f}  Holm p={adj[k]:.4f}  "
              f"{'SIG' if adj[k] < 0.05 else 'n.s.'}")

    # ---- verdict, with the pre-registered asymmetric reading -------------
    sig = adj["fisher_primary"] < 0.05
    print("\n" + "=" * 74)
    if sig:
        print("VERDICT: RATE COLLAPSE DETECTED.")
        print("  The high-acquisition mode collapses under delayed exposure.")
        print("  Syntax IS susceptible on the rate readout. The differential-")
        print("  susceptibility reading is REFUTED and ledger row 11b is")
        print("  rewritten, not merely rescoped.")
    else:
        print("VERDICT: NO RATE COLLAPSE DETECTED — and this is a BOUND, not a null.")
        print("  Excluded: FULL collapse of the high mode (power 1.000).")
        print("  NOT excluded: a HALF collapse (power 0.188; ~59 seeds/group")
        print("    would be needed). Write as BOUNDED-NOT-EXCLUDED with that")
        print("    bound stated.")
        print("  Do NOT report this as a sixth route to the syntax null: under")
        print("    the standing cross-method rule an underpowered probe that")
        print("    agrees is not corroboration.")
        print("  Ledger row 11b keeps ESTABLISHED (scoped); only its scope")
        print("    sentence changes.")
    print("=" * 74)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "primary_threshold": PRIMARY_THR,
        "secondary_threshold": SECONDARY_THR,
        "excluded_threshold": EXCLUDED_THR,
        "guard_floor": GUARD_FLOOR,
        "realised_n_delayed": len(delayed),
        "missing_runs": missing,
        "fisher": {str(k): v for k, v in results.items()},
        "excluded_counts": {"delayed": dh_x, "baseline": bh_x},
        "mannwhitney": {"U": float(u), "p_one_sided": float(p_mw)},
        "mean_difference": diff, "boot_ci": [float(lo), float(hi)],
        "holm": adj, "significant": bool(sig),
        "delayed_rows": delayed, "baseline_rows": base_rows,
    }, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
