"""Targeted power for the cross-mixture attraction divergence.

The exposure re-analysis (scripts/agreement_exposure_recheck.py) localised the
mixture disagreement to the 60.3M cell. That changes the cheapest sufficient
design from "4 seeds at every size" to "more seeds at 60M". This sizes both.

Monte Carlo, using the REALIZED within-cell SDs (not the source arm's SD --
that was the error in the original replication's power estimate).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPORT = json.load(
    open(ROOT / "logs" / "sv_agreement_probe" / "exposure_recheck.json")
)

PARAMS = {k: v for k, v in REPORT["params"].items()}
SIZES = ["11M", "23M", "60M"]
RNG = np.random.default_rng(20260727)
NSIM = 20000

per_size = REPORT["per_size"]
# realized within-cell SDs
SD_ARI = {s: per_size[s]["arith_sd"] for s in SIZES}
SD_SYN = {s: per_size[s]["syntax_sd"] for s in SIZES}
MU_SYN = {s: per_size[s]["syntax_obs"] for s in SIZES}
N_SYN = 24

out = {"realized_sds": {"arith": SD_ARI, "syntax": SD_SYN}}

# ---- (A) 60M-only cross-mixture contrast, Welch, vs number of arith seeds
# effect sizes: the observed 0.133, and a conservative 0.100
a60 = {}
for eff in (0.133, 0.100):
    a60[f"delta_{eff}"] = {}
    for n in (3, 4, 6, 8, 10, 12):
        hits = 0
        for _ in range(NSIM):
            x = RNG.normal(MU_SYN["60M"] + eff, SD_ARI["60M"], n)
            y = RNG.normal(MU_SYN["60M"], SD_SYN["60M"], N_SYN)
            if stats.ttest_ind(x, y, equal_var=False).pvalue < 0.05:
                hits += 1
        a60[f"delta_{eff}"][f"n_arith={n}"] = hits / NSIM
out["power_60M_contrast"] = a60

# ---- (B) the mixture x log10(params) interaction, adding seeds at ALL sizes
# H1: arith slope is shallower by the observed amount; simulate arith cells at
# their observed means, syntax at theirs, and test the interaction term.
mu_ari = {s: per_size[s]["arith_obs"] for s in SIZES}
logp = {s: np.log10(PARAMS[s]) for s in SIZES}


def interaction_p(n_arith_per_size):
    xs, ys, ms = [], [], []
    for s in SIZES:
        xs += [logp[s]] * N_SYN
        ys += list(RNG.normal(MU_SYN[s], SD_SYN[s], N_SYN))
        ms += [0] * N_SYN
        xs += [logp[s]] * n_arith_per_size
        ys += list(RNG.normal(mu_ari[s], SD_ARI[s], n_arith_per_size))
        ms += [1] * n_arith_per_size
    x = np.asarray(xs)
    y = np.asarray(ys)
    m = np.asarray(ms, dtype=float)
    xc = x - x.mean()
    X = np.column_stack([np.ones_like(xc), xc, m, xc * m])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    t = beta[3] / np.sqrt(cov[3, 3])
    return 2 * stats.t.sf(abs(t), dof)


inter = {}
for n in (3, 4, 6, 8):
    hits = sum(interaction_p(n) < 0.05 for _ in range(2000))
    inter[f"n_arith_per_size={n}"] = hits / 2000
out["power_interaction"] = inter

# ---- (C) cost: GPU runs required
out["runs_required"] = {
    "60M_only_to_n6": 3,
    "60M_only_to_n8": 5,
    "all_sizes_to_n4": 3,
    "all_sizes_to_n6": 9,
    "note": "arith d0 runs at 11M/23M/60M already exist at seeds 1-3",
}

path = ROOT / "logs" / "sv_agreement_probe" / "power_60M.json"
json.dump(out, open(path, "w"), indent=1)
print(json.dumps(out, indent=1))
print(f"\nwrote {path}")
