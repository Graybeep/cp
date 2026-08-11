"""Do ablation (necessity) and activation patching (sufficiency) agree about WHICH
triads reorganized? Joins the six circuit-identity triads across both methods."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

CI = Path("logs/week8_circuit_identity"); CT = Path("logs/week8_causal_trace")

def tag(seed, size): return f"seed{seed}" + ("" if size == "23M" else f"_{size}")

rows = []
for size in ("23M", "60M"):
    for seed in (1, 2, 3):
        ci = json.load(open(CI / f"{tag(seed,size)}.json"))
        ct = json.load(open(CT / f"seed{seed}_{size}.json"))
        b = ci["baseline_acc"]
        # patching sufficiency: A's heads vs C's own heads, both measured inside C
        aic = ct["A_heads_causal_in_C"]["mean_restoration_of_A_top_in_C"]
        cic = ct["A_heads_causal_in_C"]["mean_restoration_of_C_top_in_C"]
        rows.append(dict(size=size, seed=seed, deficit=b["A"] - b["B"],
                         abl_transfer=ci["verdict"]["transfer_ratio_C_over_A"],
                         abl_jaccard=ci["verdict"]["jaccard_A_top_C_top"],
                         pat_jaccard=ct["patching_overlap"]["jaccard"],
                         pat_A_in_C=aic, pat_Cown_in_C=cic,
                         pat_ratio=(aic / cic) if (aic and cic) else None))

print(f"{'size':>5}{'seed':>5}{'deficit':>9}{'ABL xfer':>10}{'ABL jac':>9}"
      f"{'PAT jac':>9}{'PAT A-in-C':>11}{'PAT ratio':>10}")
for r in rows:
    print(f"{r['size']:>5}{r['seed']:>5}{r['deficit']:>9.3f}{r['abl_transfer']:>10.2f}"
          f"{r['abl_jaccard']:>9.2f}{r['pat_jaccard']:>9.2f}{r['pat_A_in_C']:>11.3f}"
          f"{r['pat_ratio']:>10.2f}")

abl = [r["abl_transfer"] for r in rows]; pat = [r["pat_ratio"] for r in rows]
dfc = [r["deficit"] for r in rows]
print(f"\nABLATION  transfer range {min(abl):.2f}-{max(abl):.2f}  "
      f"(spread {max(abl)-min(abl):.2f}, SD {np.std(abl, ddof=1):.3f})")
print(f"PATCHING  ratio    range {min(pat):.2f}-{max(pat):.2f}  "
      f"(spread {max(pat)-min(pat):.2f}, SD {np.std(pat, ddof=1):.3f})")
r_, p_ = stats.pearsonr(abl, pat)
print(f"\nagreement between methods (n=6): pearson r={r_:.3f} p={p_:.3f}")
rs, ps = stats.spearmanr(abl, pat); print(f"                                spearman rho={rs:.3f} p={ps:.3f}")
rd, pd = stats.spearmanr(dfc, pat)
print(f"patching ratio vs deficit:      spearman rho={rd:.3f} p={pd:.3f}")
p23 = [r["pat_ratio"] for r in rows if r["size"] == "23M"]
p60 = [r["pat_ratio"] for r in rows if r["size"] == "60M"]
t, pt = stats.ttest_ind(p23, p60, equal_var=False)
print(f"patching ratio 23M {np.mean(p23):.3f}+/-{np.std(p23,ddof=1):.3f} vs "
      f"60M {np.mean(p60):.3f}+/-{np.std(p60,ddof=1):.3f}  Welch p={pt:.3f}")
json.dump(rows, open(CI / "method_agreement.json", "w"), indent=2)
print(f"\nwrote {CI}/method_agreement.json")
