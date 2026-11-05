"""Non-cosine (WSD) schedule generalization read-out.

Executes notes/wsd-schedule-preregistration.md: is there an arithmetic critical
period at d50 when the LR schedule is warmup-stable-decay instead of cosine?

Why the answer matters more than a robustness check: under cosine the LR at the
d50 onset has already fallen to 3.57e-4 (60% of peak); under WSD it is still at
peak (6.00e-4) until step 4000. If closure at d50 is mainly a decayed-LR effect,
the deficit should largely vanish under WSD. If it persists at ~0.21, the
schedule account is not the main story and §4.4 needs rewriting.

PRIMARY   : paired per-seed deficit under WSD, acc(d0) - acc(d50), n = 3.
SECONDARY : WSD deficit vs the cosine deficit (Welch on the two sets of paired
            per-seed deficits), reported WITH its MDE and read off the MDE,
            never off the p.

Null convention. The cosine reference uses the canonical Detect nulls, which at
seed 1 means the `_recheck` re-train (0.524), not the 0.506 original that the
fork arms use -- these are different trained models under the run_id identity
rule, and the Detect-side statistic is the one that takes `_recheck`. This
reproduces the published 0.214 +/- 0.055 exactly, which is the check that the
convention here is the same one the paper used.

Usage:  python scripts/wsd_deficit_readout.py
"""

from __future__ import annotations

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
OUT = Path("logs/wsd_schedule/readout.json")

WSD_D0 = "arith_delay0_exp20_wsd_size23M_seed{s}"
WSD_D50 = "arith_delay50_exp20_wsd_size23M_seed{s}"
COS_D0 = "arith_delay0_exp20_size23M_seed{s}"
# canonical Detect nulls (seed 1 = the _recheck re-train)
COS_D50 = {1: "arith_delay50_exp20_size23M_seed1_recheck",
           2: "arith_delay50_exp20_size23M_seed2",
           3: "arith_delay50_exp20_size23M_seed3"}

PRESENT_MIN_DEFICIT = 0.05     # pre-registered "deficit present" floor
HALF_COSINE = 0.107            # pre-registered "substantially reduced" line


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


def desc(x) -> dict:
    a = np.asarray(x, float)
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
            "n": int(a.size), "values": [float(v) for v in a]}


def mde(sd1: float, sd2: float, n1: int, n2: int,
        alpha: float = 0.05, power: float = 0.80) -> float:
    """Two-sample MDE at the given power (normal approximation)."""
    se = np.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    return float((stats.norm.ppf(1 - alpha / 2)
                  + stats.norm.ppf(power)) * se)


def main() -> None:
    wsd = {s: {"d0": final_row(WSD_D0.format(s=s)),
               "d50": final_row(WSD_D50.format(s=s))} for s in SEEDS}
    have = [s for s in SEEDS if wsd[s]["d0"] and wsd[s]["d50"]]
    missing = [s for s in SEEDS if s not in have]

    print("=" * 76)
    print("WSD (NON-COSINE) SCHEDULE READOUT")
    print("  notes/wsd-schedule-preregistration.md")
    print("=" * 76)
    if not have:
        sys.exit("\nABORT: no complete WSD seed pairs yet (need both d0 and "
                 "d50 at step 5000).")
    if missing:
        print(f"\n!! PARTIAL: seeds {missing} incomplete; realised n = "
              f"{len(have)}")

    # ---- cosine reference (canonical Detect convention) ------------------
    cos_d0 = [final_row(COS_D0.format(s=s))["acc"] for s in SEEDS]
    cos_d50 = [final_row(COS_D50[s])["acc"] for s in SEEDS]
    cos_def = [a - b for a, b in zip(cos_d0, cos_d50)]

    # ---- WSD ------------------------------------------------------------
    w_d0 = [wsd[s]["d0"]["acc"] for s in have]
    w_d50 = [wsd[s]["d50"]["acc"] for s in have]
    w_def = [a - b for a, b in zip(w_d0, w_d50)]
    w_ppl0 = [wsd[s]["d0"]["ppl"] for s in have]
    w_ppl50 = [wsd[s]["d50"]["ppl"] for s in have]

    print(f"\n{'schedule':<10} {'early (d0)':>18} {'delayed (d50)':>18} "
          f"{'deficit':>16}")
    print("-" * 76)
    for name, a, b, d in (("cosine", cos_d0, cos_d50, cos_def),
                          ("WSD", w_d0, w_d50, w_def)):
        da, db, dd = desc(a), desc(b), desc(d)
        print(f"{name:<10} {da['mean']:>8.3f} +/- {da['sd']:<6.3f} "
              f"{db['mean']:>8.3f} +/- {db['sd']:<6.3f} "
              f"{dd['mean']:>7.3f} +/- {dd['sd']:<6.3f}")

    print("\nper-seed deficits (the readable part at n=3):")
    print("  cosine: " + "  ".join(f"s{s}:{v:+.3f}"
                                   for s, v in zip(SEEDS, cos_def)))
    print("  WSD:    " + "  ".join(f"s{s}:{v:+.3f}"
                                   for s, v in zip(have, w_def)))

    # ---- PRIMARY: is there a deficit under WSD? --------------------------
    all_pos = all(v > 0 for v in w_def)
    wm = float(np.mean(w_def))
    present = all_pos and wm > PRESENT_MIN_DEFICIT
    t_p = stats.ttest_rel(w_d0, w_d50) if len(have) > 1 else (np.nan, np.nan)
    print(f"\n[PRIMARY] WSD paired deficit = {wm:+.3f} "
          f"(all seeds positive: {all_pos}; paired t p = {t_p[1]:.4f})")

    # ---- SECONDARY: WSD vs cosine deficit -------------------------------
    welch = stats.ttest_ind(w_def, cos_def, equal_var=False)
    m = mde(float(np.std(w_def, ddof=1)), float(np.std(cos_def, ddof=1)),
            len(w_def), len(cos_def))
    print(f"[SECONDARY] WSD deficit - cosine deficit = "
          f"{wm - float(np.mean(cos_def)):+.3f}, Welch p = {welch.pvalue:.4f}, "
          f"MDE at 80% power = {m:.3f}")

    # ---- verdict against the pre-registered rules ------------------------
    print("\n" + "=" * 76)
    if not present:
        verdict = ("DEFICIT ABOLISHED under WSD. The schedule account is "
                   "CONFIRMED on an independent schedule: closure at d50 "
                   "tracks the decayed LR, not the delay per se.")
    elif wm < HALF_COSINE:
        verdict = (f"DEFICIT SUBSTANTIALLY REDUCED ({wm:.3f} vs cosine "
                   f"{np.mean(cos_def):.3f}). Both factors contribute; report "
                   "the schedule as a major but PARTIAL cause, with the "
                   "residual attributable to something else.")
    else:
        verdict = (f"DEFICIT PERSISTS under WSD ({wm:.3f}). Closure is NOT "
                   "mainly an LR-schedule effect. Section 4.4's central claim "
                   "needs rewriting, and the cosine result must be re-read as "
                   "alignment between the decay curve and the onset.")
    print("VERDICT: " + verdict)
    print(f"\nMDE CAVEAT (pre-registered): at n=3 vs 3 the MDE is {m:.3f}, so a "
          f"non-significant\n  WSD-vs-cosine difference is 'consistent with, "
          "and not distinguishable from'.\n  Never write 'the deficit is the "
          "same under both schedules'.")
    print("=" * 76)

    print(f"\nperplexity (descriptive only; cross-schedule ppl differences are "
          f"NOT\n  evidence about the deficit, which is a within-schedule "
          f"contrast):")
    print(f"  WSD d0  {desc(w_ppl0)['mean']:.2f}   "
          f"WSD d50 {desc(w_ppl50)['mean']:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "realised_n": len(have), "missing_seeds": missing,
        "wsd": {"d0": desc(w_d0), "d50": desc(w_d50),
                "deficit": desc(w_def),
                "ppl_d0": desc(w_ppl0), "ppl_d50": desc(w_ppl50)},
        "cosine": {"d0": desc(cos_d0), "d50": desc(cos_d50),
                   "deficit": desc(cos_def)},
        "primary": {"all_seeds_positive": all_pos, "mean": wm,
                    "paired_p": float(t_p[1]), "deficit_present": bool(present)},
        "secondary": {"difference": wm - float(np.mean(cos_def)),
                      "welch_p": float(welch.pvalue), "mde_80": m},
        "verdict": verdict,
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
