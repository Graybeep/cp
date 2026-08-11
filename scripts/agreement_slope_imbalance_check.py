"""Does the n=14 arithmetic-mixture slope test survive its (3, 3, 8) seed imbalance?

The published slope (-0.207/decade, p=0.033) comes from stats.linregress over 14
model-level points with 3 seeds at 11.5M, 3 at 23.2M, and 8 at 60.3M
(scripts/agreement_60M_replication_readout.py, "secondary"). OLS with replicated
x-values is unbiased under imbalance, but its SE assumes HOMOSCEDASTIC residuals --
and here the largest-n cell is also an endpoint (max leverage) with the largest SD.
That combination is exactly when the OLS SE can be wrong, so re-estimate under
estimators that do not assume it, and report the spread of p-values.

Estimators (same data, same points, different SE / weighting assumptions):
  1. OLS                 -- as published
  2. HC3                 -- heteroscedasticity-consistent SE, no equal-variance assumption
  3. WLS 1/s^2_group     -- downweights the noisier cell
  4. group-mean OLS      -- 3 points, one per size: fully "evenly sampled" treatment
  5. stratified bootstrap -- resample seeds WITHIN size, preserving (3,3,8)
  6. balanced subsample  -- 3-of-8 at 60.3M, all C(8,3) draws: what a (3,3,3) design gives

Writes logs/sv_agreement_probe/slope_imbalance_check.{json,md}.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("logs/sv_agreement_probe")
PARAMS = {"11M": 11_536_640, "23M": 23_204_736, "60M": 60_304_000}
SIZES = ("11M", "23M", "60M")
RNG = np.random.default_rng(0)
NBOOT = 20000


def load(fname):
    models = json.loads((OUT / fname).read_text())["models"]
    acc = {}
    for rid, v in models.items():
        m = re.search(r"size(\d+M)_seed(\d+)", rid)
        if not m:
            continue
        acc.setdefault(m.group(1), {})[int(m.group(2))] = v["verb"]
    return acc


def design(d, key):
    """Model-level (x=log10 params, y) points, in size order."""
    xs, ys, gs = [], [], []
    for s in SIZES:
        for seed in sorted(d[s]):
            v = d[s][seed]
            y = (v["attraction"] - v["complexity"]) if key == "paired" else v[key]
            xs.append(math.log10(PARAMS[s]))
            ys.append(y)
            gs.append(s)
    return np.array(xs), np.array(ys), np.array(gs)


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 2
    s2 = resid @ resid / dof
    XtXi = np.linalg.inv(X.T @ X)
    se = math.sqrt(s2 * XtXi[1, 1])
    t = beta[1] / se
    return beta[1], se, float(2 * stats.t.sf(abs(t), dof)), dof, X, resid, XtXi


def hc3(x, y):
    b, _, _, dof, X, resid, XtXi = ols(x, y)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)          # leverages
    omega = (resid / (1 - h)) ** 2                     # HC3 weighting
    cov = XtXi @ (X.T * omega) @ X @ XtXi
    se = math.sqrt(cov[1, 1])
    t = b / se
    return b, se, float(2 * stats.t.sf(abs(t), dof)), h


def wls(x, y, g):
    w = np.ones_like(y)
    for s in SIZES:
        m = g == s
        v = y[m].var(ddof=1)
        w[m] = 1.0 / v
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    XtWXi = np.linalg.inv(X.T @ W @ X)
    beta = XtWXi @ X.T @ W @ y
    resid = y - X @ beta
    dof = len(y) - 2
    s2 = (resid * w @ resid) / dof
    se = math.sqrt(s2 * XtWXi[1, 1])
    t = beta[1] / se
    return beta[1], se, float(2 * stats.t.sf(abs(t), dof))


def report(name, d, key):
    x, y, g = design(d, key)
    b_ols, se_ols, p_ols, dof, _, _, _ = ols(x, y)
    b_h, se_h, p_h, lev = hc3(x, y)
    b_w, se_w, p_w = wls(x, y, g)

    # group-mean regression: one point per size, sizes equally weighted
    gx = np.array([math.log10(PARAMS[s]) for s in SIZES])
    gy = np.array([y[g == s].mean() for s in SIZES])
    gm = stats.linregress(gx, gy)

    # stratified bootstrap: resample seeds within each size, keep (3,3,8)
    idx = {s: np.flatnonzero(g == s) for s in SIZES}
    boots = np.empty(NBOOT)
    for i in range(NBOOT):
        take = np.concatenate([RNG.choice(idx[s], size=len(idx[s]), replace=True)
                               for s in SIZES])
        xb, yb = x[take], y[take]
        if xb.std() == 0:
            boots[i] = np.nan
            continue
        boots[i] = np.polyfit(xb, yb, 1)[0]
    boots = boots[~np.isnan(boots)]
    ci = np.percentile(boots, [2.5, 97.5])
    p_boot = 2 * min((boots >= 0).mean(), (boots <= 0).mean())

    # every balanced (3,3,3) subsample of the 60.3M cell
    sub = []
    for combo in itertools.combinations(idx["60M"], 3):
        take = np.concatenate([idx["11M"], idx["23M"], np.array(combo)])
        r = stats.linregress(x[take], y[take])
        sub.append((r.slope, r.pvalue))
    sub_sl = np.array([s for s, _ in sub])
    sub_p = np.array([p for _, p in sub])

    per_size = {s: {"n": int((g == s).sum()), "mean": float(y[g == s].mean()),
                    "sd": float(y[g == s].std(ddof=1)),
                    "leverage_each": float(lev[g == s][0])} for s in SIZES}

    return {
        "mixture": name, "quantity": key, "n_points": len(y),
        "per_size": per_size,
        "sd_ratio_max_over_min": max(v["sd"] for v in per_size.values())
                                 / min(v["sd"] for v in per_size.values()),
        "levene_p": float(stats.levene(*[y[g == s] for s in SIZES]).pvalue),
        "ols": {"slope": b_ols, "se": se_ols, "p": p_ols, "df": dof},
        "hc3": {"slope": b_h, "se": se_h, "p": p_h},
        "wls_inv_group_var": {"slope": b_w, "se": se_w, "p": p_w},
        "group_mean_ols": {"slope": float(gm.slope), "se": float(gm.stderr),
                           "p": float(gm.pvalue), "df": 1},
        "stratified_bootstrap": {"slope_mean": float(boots.mean()),
                                 "ci95": [float(ci[0]), float(ci[1])],
                                 "p_two_sided": float(p_boot), "n_boot": len(boots)},
        "balanced_333_subsamples": {
            "n_subsamples": len(sub_sl),
            "slope_mean": float(sub_sl.mean()),
            "slope_range": [float(sub_sl.min()), float(sub_sl.max())],
            "p_median": float(np.median(sub_p)),
            "frac_p_below_05": float((sub_p < 0.05).mean()),
        },
    }


ari = load("quartet_scale_arithmix_n8.json")
syn = load("quartet_scale.json")

rep = {"arith": {k: report("arith", ari, k)
                 for k in ("attraction", "complexity", "paired")},
       "syntax": {k: report("syntax", syn, k)
                  for k in ("attraction", "complexity", "paired")}}

json.dump(rep, open(OUT / "slope_imbalance_check.json", "w"), indent=1, default=float)

L = ["# Seed-imbalance robustness of the mixture slope tests",
     "",
     "Arithmetic mixture is (3, 3, 8) seeds across 11.5M / 23.2M / 60.3M;",
     "syntax mixture is (24, 24, 24) and serves as the balanced reference.",
     "All estimators use the same model-level points; they differ only in what",
     "they assume about residual variance and how sizes are weighted.",
     ""]
for mix in ("arith", "syntax"):
    for key in ("attraction", "complexity", "paired"):
        r = rep[mix][key]
        L += [f"## {mix} — {key} (n = {r['n_points']})", "",
              "| size | n | mean | SD | leverage/pt |",
              "|:--|--:|--:|--:|--:|"]
        for s in SIZES:
            v = r["per_size"][s]
            L.append(f"| {s} | {v['n']} | {v['mean']:+.3f} | {v['sd']:.3f} "
                     f"| {v['leverage_each']:.3f} |")
        L += ["",
              f"SD ratio max/min = {r['sd_ratio_max_over_min']:.2f}, "
              f"Levene p = {r['levene_p']:.3f}", "",
              "| estimator | slope/decade | SE | p |", "|:--|--:|--:|--:|",
              f"| OLS (published) | {r['ols']['slope']:+.3f} | {r['ols']['se']:.3f} | {r['ols']['p']:.4f} |",
              f"| HC3 robust | {r['hc3']['slope']:+.3f} | {r['hc3']['se']:.3f} | {r['hc3']['p']:.4f} |",
              f"| WLS 1/s²_group | {r['wls_inv_group_var']['slope']:+.3f} | {r['wls_inv_group_var']['se']:.3f} | {r['wls_inv_group_var']['p']:.4f} |",
              f"| group-mean OLS (df=1) | {r['group_mean_ols']['slope']:+.3f} | {r['group_mean_ols']['se']:.3f} | {r['group_mean_ols']['p']:.4f} |",
              f"| stratified bootstrap | {r['stratified_bootstrap']['slope_mean']:+.3f} | CI {r['stratified_bootstrap']['ci95'][0]:+.3f}, {r['stratified_bootstrap']['ci95'][1]:+.3f} | {r['stratified_bootstrap']['p_two_sided']:.4f} |",
              f"| balanced (3,3,3) subsamples | {r['balanced_333_subsamples']['slope_mean']:+.3f} "
              f"(range {r['balanced_333_subsamples']['slope_range'][0]:+.3f}…{r['balanced_333_subsamples']['slope_range'][1]:+.3f}) "
              f"| — | median {r['balanced_333_subsamples']['p_median']:.4f}, "
              f"{r['balanced_333_subsamples']['frac_p_below_05']:.0%} < 0.05 |",
              ""]
(OUT / "slope_imbalance_check.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
