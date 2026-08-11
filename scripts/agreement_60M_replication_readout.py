"""Read out the 60.3M mixture replication against the PRE-REGISTERED rules.

Rules are fixed in notes/agreement-60M-mixture-replication-preregistration.md and
must not be varied here. This script only applies them.

PRIMARY (the only gating test): Welch two-sample t, TWO-SIDED, alpha=0.05, on
verb.attraction at 60.3M -- arithmetic mixture (n=8) vs syntax mixture (n=24).
  (A) p<0.05 and arith>syntax -> divergence is REAL
  (B) p>=0.05               -> divergence CLOSES AS NOISE (with MDE attached)
  (C) p<0.05 and arith<syntax -> third outcome, reported separately
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("logs/sv_agreement_probe")
PARAMS = {"11M": 11_536_640, "23M": 23_204_736, "60M": 60_304_000}
SIZES = ("11M", "23M", "60M")


def load(fname):
    models = json.loads((OUT / fname).read_text())["models"]
    acc = {}
    for rid, v in models.items():
        m = re.search(r"size(\d+M)_seed(\d+)", rid)
        if not m:
            continue
        acc.setdefault(m.group(1), {})[int(m.group(2))] = v["verb"]
    return acc


ari = load("quartet_scale_arithmix_n8.json")
syn = load("quartet_scale.json")

rep = {}


def arr(d, size, key):
    return np.array([d[size][s][key] for s in sorted(d[size])])


# ---------------------------------------------------------------- validity
uni = json.loads((OUT / "quartet_scale_arithmix_n8.json").read_text())
rep["validity"] = {
    "unigram_control_attraction": uni["control_unigram"]["attraction"],
    "n_arith_60M": len(ari["60M"]),
    "n_syntax_60M": len(syn["60M"]),
    "attraction_positive_all_arith_60M": bool(
        (arr(ari, "60M", "attraction") > 0).all()
    ),
}

# ------------------------------------------------------- PRIMARY TEST
a = arr(ari, "60M", "attraction")
b = arr(syn, "60M", "attraction")
t, p = stats.ttest_ind(a, b, equal_var=False)
se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
df = se**4 / (
    (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
    + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
)
tc = stats.t.ppf(0.975, df)
d = a.mean() - b.mean()

if p < 0.05 and d > 0:
    verdict = "A_DIVERGENCE_REAL"
elif p < 0.05 and d < 0:
    verdict = "C_UNANTICIPATED_SIGN"
else:
    verdict = "B_CLOSES_AS_NOISE"

rep["primary"] = {
    "arith_60M_mean": a.mean(), "arith_60M_sd": a.std(ddof=1), "n": len(a),
    "arith_per_seed": {int(s): ari["60M"][s]["attraction"] for s in sorted(ari["60M"])},
    "syntax_60M_mean": b.mean(), "syntax_60M_sd": b.std(ddof=1), "n_syn": len(b),
    "diff": d, "ci95": [d - tc * se, d + tc * se],
    "welch_t": float(t), "df": float(df), "p_two_sided": float(p),
    "VERDICT": verdict,
    "n3_diff_for_reference": 0.1330208333333334,
}

# the n=3 subset, to show what changed
a3 = np.array([ari["60M"][s]["attraction"] for s in (1, 2, 3)])
a5 = np.array([ari["60M"][s]["attraction"] for s in (4, 5, 6, 7, 8)])
rep["what_the_new_seeds_did"] = {
    "seeds_1_3_mean": a3.mean(), "seeds_1_3_sd": a3.std(ddof=1),
    "seeds_4_8_mean": a5.mean(), "seeds_4_8_sd": a5.std(ddof=1),
    "new_minus_old": a5.mean() - a3.mean(),
    "p_old_vs_new": float(stats.ttest_ind(a3, a5, equal_var=False).pvalue),
}

# ------------------------------------------------- SECONDARY (non-gating)
sec = {}
for name, dd in (("arith", ari), ("syntax", syn)):
    xs, ys, cs = [], [], []
    for s in SIZES:
        for seed in sorted(dd[s]):
            xs.append(math.log10(PARAMS[s]))
            ys.append(dd[s][seed]["attraction"])
            cs.append(dd[s][seed]["complexity"])
    r = stats.linregress(xs, ys)
    rc_ = stats.linregress(xs, cs)
    sec[name] = {
        "attraction_slope": r.slope, "p": r.pvalue, "n": len(xs),
        "complexity_slope": rc_.slope, "complexity_p": rc_.pvalue,
        "means": {s: float(np.mean([dd[s][k]["attraction"] for k in dd[s]]))
                  for s in SIZES},
        "monotone_11_23_60": (
            np.mean([dd["11M"][k]["attraction"] for k in dd["11M"]])
            > np.mean([dd["23M"][k]["attraction"] for k in dd["23M"]])
            > np.mean([dd["60M"][k]["attraction"] for k in dd["60M"]])
        ),
    }

# pooled interaction, mixture x log10(params)
xs, ys, ms = [], [], []
for name, dd, mix in (("arith", ari, 1), ("syntax", syn, 0)):
    for s in SIZES:
        for seed in sorted(dd[s]):
            xs.append(math.log10(PARAMS[s]))
            ys.append(dd[s][seed]["attraction"])
            ms.append(mix)
x = np.array(xs); y = np.array(ys); m = np.array(ms, float)
xc = x - x.mean()
X = np.column_stack([np.ones_like(xc), xc, m, xc * m])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - X.shape[1]
cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
tint = beta[3] / math.sqrt(cov[3, 3])
sec["pooled_interaction"] = {
    "n_models": len(y),
    "estimate": float(beta[3]), "se": float(math.sqrt(cov[3, 3])),
    "t": float(tint), "p": float(2 * stats.t.sf(abs(tint), dof)),
    "previous_p_at_n3": 0.050,
}

# per-size cross-mixture contrasts
sec["per_size_contrast"] = {}
for s in SIZES:
    aa, bb = arr(ari, s, "attraction"), arr(syn, s, "attraction")
    tt, pp = stats.ttest_ind(aa, bb, equal_var=False)
    sec["per_size_contrast"][s] = {
        "arith": aa.mean(), "syntax": bb.mean(), "diff": aa.mean() - bb.mean(),
        "p": float(pp), "n_arith": len(aa),
    }

# controlled mismatch at 60.3M (the "below chance at smallest size only" claim)
mm = arr(ari, "60M", "balanced_mismatch")
tm, pm = stats.ttest_1samp(mm, 0.5)
sec["mismatch_60M_vs_chance"] = {
    "mean": mm.mean(), "sd": mm.std(ddof=1), "t": float(tm), "p": float(pm)}
rep["secondary"] = sec

path = OUT / "replication_60M_readout.json"
json.dump(rep, open(path, "w"), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
print(f"\nwrote {path}")
