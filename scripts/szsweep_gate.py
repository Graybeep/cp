"""Syntax LD size-sweep — pre-registered confirm/disconfirm gate.

Reads the design from notes/week5-syntax-size-sweep-preregistration.md and
executes it verbatim on the 72 completed runs (3 sizes x 24 seeds, delay-0,
exp10, pool v1, step-5000 final checkpoint, continuation-scored LD).

Pipeline:
  1. Ensure per-item win arrays exist for every final ckpt (reuses the tested
     evaluate() from gap_region_interaction.py; saves to logs/szsweep_eval/items).
  2. Per run: LD interaction, local interaction, and the with_gap/no_gap LD
     components (for the validity guard).
  3. Recompute mu_low, sigma_low from the SMALLEST model (11M) low cluster ->
     derived threshold mu_low + 3 sigma_low (reported; decision uses the
     pre-committed robustness band {0.55, 0.56, 0.57} regardless).
  4. P(high) per size at each threshold; Cochran-Armitage one-sided trend test.
  5. Compound gate: CA p<0.05 at ALL thresholds AND monotone non-decreasing
     AND top-size P(high) >= 0.40.  Plus mean-LD-vs-size and Wilson 95% CIs.

Outputs a human report to stdout and a machine record to
logs/szsweep_eval/gate_result.json.  Analysis-outputs-in-logs rule.

Run:  python scripts/szsweep_gate.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import CHECKPOINT_DIR, DATA_DIR  # noqa: E402
from scripts.gap_region_interaction import evaluate  # noqa: E402

SIZES = ["11M", "23M", "60M"]          # ordered small -> large
SEEDS = list(range(1, 25))             # 24 per size
THRESHOLDS = [0.55, 0.56, 0.57]        # pre-committed robustness band
FLOOR = 0.40                           # effect-size floor on top-size P(high)
GUARD_FLOOR = 0.15                     # neither LD paradigm may collapse to prior
ITEMS_DIR = Path("logs/szsweep_eval/items")
OUT_JSON = Path("logs/szsweep_eval/gate_result.json")

LD_WITH = "wh_vs_that_with_gap_long_distance"
LD_NO = "wh_vs_that_no_gap_long_distance"
LOC_WITH = "wh_vs_that_with_gap"
LOC_NO = "wh_vs_that_no_gap"


def run_id(size: str, seed: int) -> str:
    return f"syntax_delay0_exp10_size{size}_seed{seed}_szsweep"


def mean(xs: list[int]) -> float:
    return sum(xs) / len(xs)


def load_run(size: str, seed: int, tok, device: str) -> dict:
    """Return per-run metrics, computing+caching item arrays if missing."""
    rid = run_id(size, seed)
    item_file = ITEMS_DIR / f"{rid}_step0005000_items.json"
    if not item_file.exists():
        evaluate(rid, tok, device, every=0, model_cache={},
                 save_items=ITEMS_DIR)
    items = json.loads(item_file.read_text())["items"]
    with_gap = mean(items[LD_WITH])
    no_gap = mean(items[LD_NO])
    return {
        "size": size, "seed": seed, "run_id": rid,
        "ld": (with_gap + no_gap) / 2,
        "ld_with_gap": with_gap, "ld_no_gap": no_gap,
        "local": (mean(items[LOC_WITH]) + mean(items[LOC_NO])) / 2,
    }


def guarded_high(row: dict, thr: float) -> bool:
    """High = LD interaction >= thr AND both paradigms genuinely contribute
    (neither collapsed to the unigram/prior floor)."""
    return (row["ld"] >= thr
            and row["ld_with_gap"] >= GUARD_FLOOR
            and row["ld_no_gap"] >= GUARD_FLOOR)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cochran_armitage(counts: list[int], ns: list[int],
                     scores: list[float]) -> tuple[float, float, float]:
    """One-sided Cochran-Armitage trend test for positive slope.

    Returns (z, p_one_sided, slope_sign). counts[i]=#high, ns[i]=#seeds at
    dose scores[i]. Standard large-sample statistic (Agresti).
    """
    N = sum(ns)
    R = sum(counts)                       # total "successes"
    p_bar = R / N
    s_bar = sum(n * s for n, s in zip(ns, scores)) / N
    # numerator: sum_i count_i * (s_i - s_bar)
    T = sum(c * (s - s_bar) for c, s in zip(counts, scores))
    var = p_bar * (1 - p_bar) * sum(n * (s - s_bar) ** 2
                                    for n, s in zip(ns, scores))
    if var <= 0:
        return (0.0, 1.0, 0.0)
    z = T / math.sqrt(var)
    p_one = 1 - normal_cdf(z)             # positive-slope (upper tail)
    return (z, p_one, math.copysign(1.0, T))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # guard: all 72 final checkpoints must exist
    missing = [run_id(s, sd) for s in SIZES for sd in SEEDS
               if not (CHECKPOINT_DIR / run_id(s, sd) / "step_0005000.pt").exists()]
    if missing:
        sys.exit(f"ABORT: {len(missing)} final ckpts missing, e.g. {missing[:3]}")

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)

    rows = {s: [load_run(s, sd, tok, args.device) for sd in SEEDS]
            for s in SIZES}

    # --- derived threshold from the smallest model's low cluster ----------
    ld_11 = sorted(r["ld"] for r in rows["11M"])
    # low cluster = 11M seeds below the eyeball high band (0.56); if none are
    # high, this is all 24. Robust to a stray high seed inflating sigma.
    low = [x for x in ld_11 if x < 0.56]
    mu_low = mean(low)
    sd_low = math.sqrt(sum((x - mu_low) ** 2 for x in low) / (len(low) - 1))
    derived_thr = mu_low + 3 * sd_low

    print("=" * 74)
    print("SYNTAX LD SIZE-SWEEP — PRE-REGISTERED GATE")
    print("  notes/week5-syntax-size-sweep-preregistration.md")
    print("=" * 74)
    print(f"\n11M low cluster (n={len(low)} of 24): "
          f"mu_low={mu_low:.4f}  sigma_low={sd_low:.4f}")
    print(f"derived threshold mu_low + 3*sigma_low = {derived_thr:.4f}  "
          f"(decision band stays {THRESHOLDS})")

    # --- per-size LD distribution -----------------------------------------
    print("\nMean LD interaction by size (secondary/mechanism descriptor):")
    for s in SIZES:
        lds = [r["ld"] for r in rows[s]]
        m = mean(lds)
        sd = math.sqrt(sum((x - m) ** 2 for x in lds) / (len(lds) - 1))
        print(f"  {s:>4}: mean={m:.4f}  sd={sd:.4f}  "
              f"min={min(lds):.3f}  max={max(lds):.3f}")

    # --- P(high) and CA trend at each threshold ---------------------------
    scores = [0.0, 1.0, 2.0]              # equal-ordinal dose (log-spaced sizes)
    band = {}
    print("\nP(high) by size and Cochran-Armitage one-sided trend:")
    for thr in THRESHOLDS:
        counts = [sum(guarded_high(r, thr) for r in rows[s]) for s in SIZES]
        ns = [len(rows[s]) for s in SIZES]
        rates = [c / n for c, n in zip(counts, ns)]
        z, p, sign = cochran_armitage(counts, ns, scores)
        monotone = all(rates[i] <= rates[i + 1] + 1e-9
                       for i in range(len(rates) - 1))
        cis = [wilson(c, n) for c, n in zip(counts, ns)]
        band[thr] = {"counts": counts, "ns": ns, "rates": rates,
                     "z": z, "p_one_sided": p, "monotone": monotone,
                     "top_rate": rates[-1], "wilson": cis}
        rate_str = "  ".join(
            f"{s}={c}/{n}={r:.3f}[{lo:.2f},{hi:.2f}]"
            for s, c, n, r, (lo, hi) in zip(SIZES, counts, ns, rates, cis))
        print(f"  thr={thr}: {rate_str}")
        print(f"           CA z={z:+.3f} p1={p:.4f} "
              f"monotone={monotone} top>={FLOOR}:{rates[-1] >= FLOOR}")

    # --- compound gate ----------------------------------------------------
    ca_all = all(band[t]["p_one_sided"] < 0.05 for t in THRESHOLDS)
    mono_all = all(band[t]["monotone"] for t in THRESHOLDS)
    floor_all = all(band[t]["top_rate"] >= FLOOR for t in THRESHOLDS)
    confirm = ca_all and mono_all and floor_all

    print("\n" + "=" * 74)
    print("COMPOUND GATE (all must hold to CONFIRM reposition):")
    print(f"  [{'PASS' if ca_all else 'FAIL'}] CA one-sided p<0.05 at ALL "
          f"thresholds {THRESHOLDS}")
    print(f"  [{'PASS' if mono_all else 'FAIL'}] rate monotone "
          f"non-decreasing at ALL thresholds")
    print(f"  [{'PASS' if floor_all else 'FAIL'}] top-size P(high) >= "
          f"{FLOOR} at ALL thresholds")
    verdict = "CONFIRM reposition" if confirm else "DISCONFIRM reposition"
    print(f"\n  VERDICT: {verdict}")
    if confirm:
        print("  -> syntax LD earns a delay x size slot in the Weeks 5-7 grid;")
        print("     size axis repositioned to characterize the mode's scaling.")
    else:
        # classify the disconfirm flavor
        if any(band[t]["p_one_sided"] < 0.05 for t in THRESHOLDS) and not floor_all:
            flavor = "significant but sub-floor -> 'weak size sensitivity'"
        elif not any(band[t]["monotone"] for t in THRESHOLDS):
            flavor = "non-monotone / noise (H0 not rejected)"
        else:
            flavor = "flat rate (size-independent high mode)"
        print(f"  -> {flavor}; deprioritize STANDS. Syntax LD dropped from the")
        print("     costly delay x size grid; arithmetic carries Detect.")
    print("=" * 74)

    # --- reproduce-the-top-draw reminder ----------------------------------
    top_rows = sorted(rows[SIZES[-1]], key=lambda r: -r["ld"])
    tr = top_rows[0]
    print(f"\nHighest {SIZES[-1]} seed: seed{tr['seed']} LD={tr['ld']:.3f} "
          f"(with_gap={tr['ld_with_gap']:.3f} no_gap={tr['ld_no_gap']:.3f})")
    print("  Pre-reg rule: re-run this seed once before any headline claim "
          "uses it.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "sizes": SIZES, "thresholds": THRESHOLDS, "floor": FLOOR,
        "guard_floor": GUARD_FLOOR,
        "mu_low": mu_low, "sigma_low": sd_low, "derived_threshold": derived_thr,
        "band": {str(k): v for k, v in band.items()},
        "confirm": confirm, "verdict": verdict,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
