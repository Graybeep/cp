"""Second-mixture replication test for the attraction x scale dissociation.

WHY. The headline result of the agreement-attraction note — attraction falls -0.316 per
decade of parameters while complexity stays flat (n=72, p=2.3e-17) — comes entirely from
ONE training mixture (10% syntax-enriched). A p-value that small speaks to seed and item
variance at that mixture; it says nothing about mixture variance. The arithmetic-mixture
d0 checkpoints exist at all three sizes (seeds 1-3), so the control is evaluation-only.

Compares:
  logs/sv_agreement_probe/quartet_scale.json          syntax mixture, 3 sizes x 24 seeds
  logs/sv_agreement_probe/quartet_scale_arithmix.json arith  mixture, 3 sizes x 3  seeds

PRE-REGISTERED READ (fixed before the arith run):
  CONFIRM  attraction slope vs log10(params) negative at p<0.05, ordering 11M>23M>60M,
           complexity slope non-significant.
  BREAK    attraction flat/positive, or complexity moves comparably (trend is generic).

Run: python scripts/agreement_mixture_replication.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = Path("logs/sv_agreement_probe")
PARAMS = {"11M": 11.5e6, "23M": 23.2e6, "60M": 60.3e6}
SIZES = ("11M", "23M", "60M")


def load(fname: str) -> dict[str, dict[str, np.ndarray]]:
    """-> {size: {metric: array over seeds}}."""
    models = json.loads((OUT / fname).read_text())["models"]
    acc: dict[str, list] = {}
    for rid, v in models.items():
        for s in SIZES:
            if f"size{s}_" in rid:
                acc.setdefault(s, []).append(
                    (v["verb"]["attraction"], v["verb"]["complexity"],
                     v["verb"]["balanced_mismatch"]))
    return {s: dict(zip(("attraction", "complexity", "mismatch"),
                        np.array(acc[s]).T)) for s in SIZES}


def welch(a: np.ndarray, b: np.ndarray):
    """Difference a-b with Welch CI and p."""
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                    + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    tc = stats.t.ppf(0.975, df)
    return d, d - tc * se, d + tc * se, float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def power(gap: float, sd: float, n: int) -> float:
    """Two-sample t power at alpha=.05, equal n per arm."""
    ncp = (gap / sd) * np.sqrt(n / 2)
    df, crit = 2 * n - 2, stats.t.ppf(0.975, 2 * n - 2)
    return float(1 - stats.nct.cdf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def slope(D, metric):
    x = np.concatenate([[np.log10(PARAMS[s])] * len(D[s][metric]) for s in SIZES])
    y = np.concatenate([D[s][metric] for s in SIZES])
    r = stats.linregress(x, y)
    return float(r.slope), float(r.rvalue), float(r.pvalue)


def main():
    A, S = load("quartet_scale_arithmix.json"), load("quartet_scale.json")
    report = {}

    for name, D in (("syntax-mix (10% syntax-enriched)", S),
                    ("arith-mix (20% arithmetic)", A)):
        print(f"\n=== {name}  n={len(D['11M']['attraction'])}/size ===")
        for s in SIZES:
            d = D[s]
            print(f"  {s}: attraction {d['attraction'].mean():+.3f} "
                  f"+/- {d['attraction'].std(ddof=1):.3f} | "
                  f"complexity {d['complexity'].mean():+.3f} | "
                  f"mismatch {d['mismatch'].mean():.3f}")
        for m in ("attraction", "complexity"):
            sl, r, p = slope(D, m)
            print(f"  {m:11s} slope {sl:+.3f}/decade  r={r:+.3f}  p={p:.3g}")
        g, lo, hi, p = welch(D["11M"]["attraction"], D["60M"]["attraction"])
        mono = (D["11M"]["attraction"].mean() > D["23M"]["attraction"].mean()
                > D["60M"]["attraction"].mean())
        print(f"  11M-60M gap {g:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  p={p:.3g}"
              f"   monotone 11>23>60: {mono}")
        key = "arith" if "arith" in name else "syntax"
        report[key] = {"gap": g, "ci": [lo, hi], "gap_p": p, "monotone": bool(mono),
                       "attraction_slope": slope(D, "attraction"),
                       "complexity_slope": slope(D, "complexity"),
                       "per_size": {s: {m: [float(D[s][m].mean()),
                                            float(D[s][m].std(ddof=1))]
                                        for m in ("attraction", "complexity", "mismatch")}
                                    for s in SIZES}}

    # ---- is the arith result informative, or just underpowered? ----
    syn_gap = report["syntax"]["gap"]
    lo, hi = report["arith"]["ci"]
    print("\n=== is the failure to replicate INFORMATIVE? ===")
    print(f"  arith CI [{lo:+.3f},{hi:+.3f}] contains 0: {lo <= 0 <= hi}; "
          f"contains the syntax gap {syn_gap:+.3f}: {lo <= syn_gap <= hi}")
    sd_s = np.mean([S[s]["attraction"].std(ddof=1) for s in ("11M", "60M")])
    sd_a = np.mean([A[s]["attraction"].std(ddof=1) for s in ("11M", "60M")])
    print(f"  within-size SD: syntax {sd_s:.3f}  arith {sd_a:.3f} "
          f"({sd_a / sd_s:.2f}x) <- why the pre-reg power estimate was optimistic")
    print(f"  power at n=3 for a {syn_gap:.3f} gap: pre-reg (SD {sd_s:.3f}) "
          f"{power(syn_gap, sd_s, 3):.2f}  realized (SD {sd_a:.3f}) "
          f"{power(syn_gap, sd_a, 3):.2f}")
    obs = report["arith"]["gap"]
    print(f"  power at n=3 for the OBSERVED {obs:.3f} gap: {power(obs, sd_a, 3):.2f}")
    need = {g: next(n for n in range(3, 400) if power(g, sd_a, n) >= 0.8)
            for g in (syn_gap, obs)}
    print(f"  seeds/size for 80% power: {need[syn_gap]} (on {syn_gap:.3f}), "
          f"{need[obs]} (on {obs:.3f})")

    # ---- does the SLOPE actually differ between mixtures? ----
    X, Y, M = [], [], []
    for D, mix in ((A, 1), (S, 0)):
        for s in SIZES:
            X += [np.log10(PARAMS[s])] * len(D[s]["attraction"])
            Y += list(D[s]["attraction"])
            M += [mix] * len(D[s]["attraction"])
    X, Y, M = np.array(X), np.array(Y), np.array(M)
    Xc = X - X.mean()
    Z = np.column_stack([np.ones(len(X)), Xc, M, Xc * M])
    b, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    res = Y - Z @ b
    dof = len(X) - Z.shape[1]
    sb = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z) * (res @ res / dof)))
    print("\n=== mixture x scale interaction (pooled 81 models) ===")
    inter = {}
    for nm, bb, ss in zip(("intercept", "log10(params)", "mixture", "logP x mixture"), b, sb):
        p = float(2 * stats.t.sf(abs(bb / ss), dof))
        inter[nm] = [float(bb), float(ss), p]
        print(f"  {nm:15s} {bb:+.4f} +/- {ss:.4f}  p={p:.4g}")

    print("\n=== per-size mixture difference (where the divergence sits) ===")
    per_size = {}
    for s in SIZES:
        d, lo_, hi_, p = welch(A[s]["attraction"], S[s]["attraction"])
        per_size[s] = [d, lo_, hi_, p]
        print(f"  {s}: arith {A[s]['attraction'].mean():+.3f} vs syntax "
              f"{S[s]['attraction'].mean():+.3f}  diff {d:+.3f}  p={p:.3g}")

    report["interaction"] = inter
    report["per_size_mixture_diff"] = per_size
    report["power"] = {"sd_syntax": float(sd_s), "sd_arith": float(sd_a),
                       "prereg_power": power(syn_gap, sd_s, 3),
                       "realized_power": power(syn_gap, sd_a, 3),
                       "power_observed_gap": power(obs, sd_a, 3),
                       "seeds_for_80pct": {str(round(k, 3)): v for k, v in need.items()}}
    (OUT / "mixture_replication.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}/mixture_replication.json")


if __name__ == "__main__":
    main()
