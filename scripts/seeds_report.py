"""Seeds 2-3 replication report for syntax — applies the PRE-REGISTERED
read (fixed 2026-07-17 ~17:30, before any seed-2/3 eval existed; see
memory week3-pilot-results):

1. DIP-SHAPE (local): seed shows the dip iff local(d25) is the strict
   minimum of {d25,d50,d75} AND local(d0)-local(d25) >= 0.032.
   3/3 seeds -> real second finding; 2/3 -> inconclusive; 0-1/3 -> noise.
2. LOCAL AMBIGUITY RESOLUTION: REAL gap iff mean-over-seeds drop >= 0.032
   at d75 AND per-seed McNemar (both paradigms) p < 0.05 in >=2/3 seeds,
   consistent direction; clustered-flat iff all mean deltas < 0.032;
   else ambiguous -> grid decides.
3. LD CONFIRMATION: step-like gap replicates iff every seed has
   LD(d0)-LD(dX) >= 0.032 for all delays and within-seed spread among
   d25/50/75 < 0.032.

Prereq: per-item files for all 12 runs (3 seeds x 4 delays), produced by
gap_region_interaction.py --save-items logs/pilot_items (final ckpt).
This script runs that automatically for any missing run.

Run:  python scripts/seeds_report.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reopen.config import LOG_DIR  # noqa: E402
from reopen.incremental import load_json_valid  # noqa: E402

ITEMS = LOG_DIR / "pilot_items"
SEEDS = (1, 2, 3)
DELAYS = (0, 25, 50, 75)
BAR = 0.032
LIC = {"local": ("wh_vs_that_with_gap", "wh_vs_that_no_gap"),
       "long_distance": ("wh_vs_that_with_gap_long_distance",
                         "wh_vs_that_no_gap_long_distance")}


def run_id(seed: int, delay: int) -> str:
    return f"syntax_delay{delay}_exp10_size23M_seed{seed}"


def items_for(seed: int, delay: int) -> dict | None:
    hits = sorted(ITEMS.glob(f"{run_id(seed, delay)}_step*_items.json"))
    return load_json_valid(hits[-1]) if hits else None


def mcnemar_p(a: list[int], b: list[int]) -> float:
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    return 1.0 if x + y == 0 else sps.binomtest(x, x + y, 0.5).pvalue


def main() -> None:
    missing = [(s, d) for s in SEEDS for d in DELAYS
               if items_for(s, d) is None]
    for s, d in missing:
        print(f"scoring {run_id(s, d)} (final checkpoint) ...", flush=True)
        subprocess.run(
            [sys.executable,
             str(Path(__file__).parent / "gap_region_interaction.py"),
             run_id(s, d), "--save-items", str(ITEMS)],
            check=True, capture_output=True)

    data = {(s, d): items_for(s, d)["items"] for s in SEEDS for d in DELAYS}
    inter = {}
    for (s, d), it in data.items():
        for dist, (pg, png) in LIC.items():
            inter[(s, d, dist)] = (sum(it[pg]) / len(it[pg])
                                   + sum(it[png]) / len(it[png])) / 2

    for dist in LIC:
        print(f"\n===== {dist} (continuation interaction, final ckpt)")
        print("delay:   " + "  ".join(f"d{d:<4}" for d in DELAYS))
        for s in SEEDS:
            print(f"seed {s}:  " + "  ".join(
                f"{inter[(s, d, dist)]:.3f}" for d in DELAYS))
        means = {d: sum(inter[(s, d, dist)] for s in SEEDS) / len(SEEDS)
                 for d in DELAYS}
        print("mean:    " + "  ".join(f"{means[d]:.3f}" for d in DELAYS))

    # 1. dip-shape replication (local)
    print("\n-- pre-registered call 1: dip-shape replication (local)")
    dips = 0
    for s in SEEDS:
        v = {d: inter[(s, d, "local")] for d in DELAYS}
        has = (v[25] < v[50] and v[25] < v[75]
               and v[0] - v[25] >= BAR)
        dips += has
        print(f"  seed {s}: d25 strict min={v[25] < v[50] and v[25] < v[75]}"
              f", d0-d25={v[0] - v[25]:+.3f} -> "
              f"{'DIP' if has else 'no dip'}")
    call1 = ("REAL second finding (3/3)" if dips == 3 else
             "INCONCLUSIVE (2/3) — more seeds" if dips == 2 else
             f"seed-1 noise ({dips}/3) — drop the dip narrative")
    print(f"  verdict: {call1}")

    # 2. local ambiguity resolution
    print("\n-- pre-registered call 2: local delay gap")
    means_l = {d: sum(inter[(s, d, "local")] for s in SEEDS) / 3
               for d in DELAYS}
    sig = {d: 0 for d in (25, 50, 75)}
    for s in SEEDS:
        for d in (25, 50, 75):
            ps = [mcnemar_p(data[(s, 0)][p], data[(s, d)][p])
                  for p in LIC["local"]]
            drop = inter[(s, 0, "local")] - inter[(s, d, "local")]
            if max(ps) < 0.05 and drop > 0:
                sig[d] += 1
    for d in (25, 50, 75):
        print(f"  d{d}: mean drop={means_l[0] - means_l[d]:+.3f}  "
              f"McNemar sig (both paradigms, drop>0) in {sig[d]}/3 seeds")
    if means_l[0] - means_l[75] >= BAR and sig[75] >= 2:
        call2 = "REAL local delay gap"
    elif all(means_l[0] - means_l[d] < BAR for d in (25, 50, 75)):
        call2 = "clustered-flat (no local gap at this resolution)"
    else:
        call2 = "STILL AMBIGUOUS — decide with grid"
    print(f"  verdict: {call2}")

    # 3. LD confirmation
    print("\n-- pre-registered call 3: LD step-like gap replication")
    okall = True
    for s in SEEDS:
        v = {d: inter[(s, d, "long_distance")] for d in DELAYS}
        drops_ok = all(v[0] - v[d] >= BAR for d in (25, 50, 75))
        spread = max(v[d] for d in (25, 50, 75)) - min(v[d]
                                                       for d in (25, 50, 75))
        flat_ok = spread < BAR
        okall &= (drops_ok and flat_ok)
        print(f"  seed {s}: all drops>=bar={drops_ok}, "
              f"spread among delays={spread:.3f} (<{BAR}={flat_ok})")
    print(f"  verdict: {'REPLICATES — step-like LD gap confirmed' if okall else 'does NOT fully replicate — inspect per-seed pattern'}")


if __name__ == "__main__":
    main()
