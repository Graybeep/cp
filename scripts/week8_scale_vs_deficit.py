"""Is the 60M 'reorganization' a SCALE effect, or is size confounded with deficit depth?

Pools all six circuit-identity triads (23M x3, 60M x3) and asks whether the two
size groups are separable in deficit -- if their deficit ranges do not overlap,
size and deficit are collinear by construction and no per-point comparison can
attribute reorganization to scale.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--tag-suffix", default="",
                help="'_cleaneval' reads the held-out-draw reports")
args = ap.parse_args()

OUT = Path("logs/week8_circuit_identity")
rows = []
for size in ("23M", "60M"):
    for seed in (1, 2, 3):
        tag = (f"seed{seed}" + ("" if size == "23M" else f"_{size}")
               + args.tag_suffix)
        r = json.load(open(OUT / f"{tag}.json"))
        b = r["baseline_acc"]
        rows.append(dict(size=size, seed=seed, A=b["A"], B=b["B"], C=b["C"],
                         deficit=b["A"] - b["B"],
                         transfer=r["verdict"]["transfer_ratio_C_over_A"],
                         control=r["verdict"]["control_ratio_Cheads_over_Aheads"],
                         jaccard=r["verdict"]["jaccard_A_top_C_top"]))
rows.sort(key=lambda d: d["deficit"])

print(f"{'size':>5} {'seed':>4} {'deficit':>8} {'transfer':>9} {'control':>8} {'jacc':>6}")
for d in rows:
    print(f"{d['size']:>5} {d['seed']:>4} {d['deficit']:>8.3f} {d['transfer']:>9.2f} "
          f"{d['control']:>8.2f} {d['jaccard']:>6.2f}")

t23 = [d["transfer"] for d in rows if d["size"] == "23M"]
t60 = [d["transfer"] for d in rows if d["size"] == "60M"]
d23 = [d["deficit"] for d in rows if d["size"] == "23M"]
d60 = [d["deficit"] for d in rows if d["size"] == "60M"]
print(f"\ntransfer  23M {np.mean(t23):.3f} +/- {np.std(t23, ddof=1):.3f}   "
      f"60M {np.mean(t60):.3f} +/- {np.std(t60, ddof=1):.3f}")
print(f"deficit   23M {np.mean(d23):.3f} [{min(d23):.3f},{max(d23):.3f}]   "
      f"60M {np.mean(d60):.3f} [{min(d60):.3f},{max(d60):.3f}]")

overlap = not (max(d23) < min(d60) or max(d60) < min(d23))
inside = min(d23) < min(d60) and max(d60) < max(d23)
print(f"\ndeficit ranges overlap: {overlap}   60M range nested inside 23M range: {inside}")

rho, p = stats.spearmanr([d["deficit"] for d in rows], [d["transfer"] for d in rows])
print(f"Spearman deficit vs transfer (all 6): rho={rho:.3f} p={p:.4f}")
rho6, p6 = stats.spearmanr(d60, t60)
print(f"Spearman within 60M (n=3):            rho={rho6:.3f} p={p6:.4f}")
t, pt = stats.ttest_ind(t23, t60, equal_var=False)
print(f"Welch transfer 23M vs 60M:            t={t:.3f} p={pt:.4f}")

# partial-out attempt: predict each 60M point from the 23M deficit->transfer line
sl, ic, r, pv, se = stats.linregress(d23, t23)
pred = [sl * d + ic for d in d60]
print(f"\n23M line: transfer = {sl:.2f}*deficit + {ic:.2f} (r={r:.2f}, p={pv:.3f}) "
      f"-- EXTRAPOLATED into a deficit range 23M never occupies")
for d, o, p_ in zip(d60, t60, pred):
    print(f"  60M deficit {d:.3f}: observed {o:.2f}  vs 23M-line {p_:.2f}  resid {o-p_:+.2f}")

json.dump({"rows": rows, "spearman_all": [rho, p], "welch": [t, pt],
           "deficit_overlap": overlap, "nested": inside},
          open(OUT / f"scale_vs_deficit{args.tag_suffix}.json", "w"), indent=2)
print(f"\nwrote {OUT}/scale_vs_deficit.json")
