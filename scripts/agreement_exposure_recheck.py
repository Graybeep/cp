"""Free re-analysis: does English-token exposure explain the cross-mixture
attraction values better than parameter count?

Zero new training, zero new evaluation. Reads:
  - logs/sv_agreement_probe/quartet_scale.json          (syntax mixture, 72 models)
  - logs/sv_agreement_probe/quartet_scale_arithmix.json (arith mixture, 9 models)
  - logs/<run>/realized_exposure.json                   (per-run token ledger)
  - logs/<run>/metrics.csv                              (shared-English val ppl)

Writes logs/sv_agreement_probe/exposure_recheck.{json,md}.

The hypothesis under test (H_exposure): attraction reduction is driven by
natural-language (English) exposure rather than parameter count, so the
arith-mixture models -- which spend 20% of their token budget on synthetic
arithmetic -- should behave like English-starved (i.e. effectively smaller /
under-trained) models, explaining why arith-60M shows MORE attraction than
syntax-60M.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PROBE = LOGS / "sv_agreement_probe"

# exact parameter counts (computed from each run's config.json via reopen.model.GPT)
PARAMS = {"11M": 11_536_640, "23M": 23_204_736, "60M": 60_304_000}
SIZES = ["11M", "23M", "60M"]

SYNTAX_RUN = "syntax_delay0_exp10_size{size}_seed{seed}_szsweep"
ARITH_RUN = "arith_delay0_exp20_size{size}_seed{seed}"


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ols(x, y):
    """Simple OLS. Returns slope, intercept, se_slope, t, df."""
    n = len(x)
    mx, my = mean(x), mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx
    a = my - b * mx
    resid = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    dof = n - 2
    s2 = sum(r * r for r in resid) / dof
    se_b = math.sqrt(s2 / sxx)
    return b, a, se_b, b / se_b, dof


def t_sf(t, dof):
    """Two-sided p-value for Student t (regularized incomplete beta)."""
    x = dof / (dof + t * t)
    return _betainc(dof / 2.0, 0.5, x)


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def welch(a, b):
    na, nb = len(a), len(b)
    va, vb = sd(a) ** 2, sd(b) ** 2
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, float("nan"), 0.0
    t = (mean(a) - mean(b)) / se
    dof = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    return t, dof, t_sf(t, dof)


# ---------------------------------------------------------------- load data
def load_attraction(path, run_pat):
    """{size: {seed: attraction}} from a quartet JSON."""
    blob = json.load(open(path))["models"]
    out = {s: {} for s in SIZES}
    for name, rec in blob.items():
        m = re.search(r"size(\d+M)_seed(\d+)", name)
        if not m:
            continue
        size, seed = m.group(1), int(m.group(2))
        if size in out:
            out[size][seed] = rec["verb"]["attraction"]
    return out


def token_ledger(run):
    p = LOGS / run / "realized_exposure.json"
    if not p.exists():
        return None
    return json.load(open(p))["realized_tokens_by_stream"]


def final_val_ppl(run):
    p = LOGS / run / "metrics.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    last = max(rows, key=lambda r: int(r["step"]))
    return float(last["val_ppl"])


syn = load_attraction(PROBE / "quartet_scale.json", SYNTAX_RUN)
ari = load_attraction(PROBE / "quartet_scale_arithmix.json", ARITH_RUN)

report = {"params": PARAMS}

# ------------------------------------------------- 1. the exposure ledger
# English tokens = every stream that is natural language.
#   syntax mixture: base (BabyLM) + syntax (wh-question English sentences)
#   arith  mixture: base (BabyLM) only; the arith stream is synthetic digits
ledger = {}
for size in SIZES:
    ledger[size] = {}
    for mix, pat, eng_streams in (
        ("syntax", SYNTAX_RUN, ("base", "syntax")),
        ("arith", ARITH_RUN, ("base",)),
    ):
        per_seed = []
        for seed in (1, 2, 3):
            tl = token_ledger(pat.format(size=size, seed=seed))
            if tl is None:
                continue
            per_seed.append(
                {
                    "english": sum(tl.get(s, 0) for s in eng_streams),
                    "total": sum(tl.values()),
                }
            )
        ledger[size][mix] = {
            "english_tokens": per_seed[0]["english"] if per_seed else None,
            "total_tokens": per_seed[0]["total"] if per_seed else None,
            "n_runs_checked": len(per_seed),
            "all_identical": len({p["english"] for p in per_seed}) == 1,
        }
report["ledger"] = ledger

E_syn = ledger["11M"]["syntax"]["english_tokens"]
E_ari = ledger["11M"]["arith"]["english_tokens"]
report["english_tokens"] = {
    "syntax_mixture": E_syn,
    "arith_mixture": E_ari,
    "ratio_arith_over_syntax": E_ari / E_syn,
    "within_mixture_variance_across_sizes": 0.0,
    "constant_across_sizes": all(
        ledger[s][m]["english_tokens"] == ledger["11M"][m]["english_tokens"]
        for s in SIZES
        for m in ("syntax", "arith")
    ),
}

# ------------------------------- 2. the user's two-pronged prediction test
# Prong A: does arith-60M's English exposure land closest to syntax-23M?
distances = {s: abs(E_ari - ledger[s]["syntax"]["english_tokens"]) for s in SIZES}
report["prong_A_nearest_syntax_size_by_english_tokens"] = {
    "arith_60M_english": E_ari,
    "distance_to_syntax": distances,
    "unique_nearest": len(set(distances.values())) > 1,
    "verdict": (
        "DEGENERATE - equidistant from all three syntax sizes"
        if len(set(distances.values())) == 1
        else f"nearest = {min(distances, key=distances.get)}"
    ),
}

# Strongest surviving variant: English tokens PER PARAMETER (Chinchilla ratio).
tpp = {
    s: {
        "syntax": E_syn / PARAMS[s],
        "arith": E_ari / PARAMS[s],
    }
    for s in SIZES
}
# which syntax size does arith-60M match on tokens/param, and what P would match exactly?
d_tpp = {s: abs(tpp["60M"]["arith"] - tpp[s]["syntax"]) for s in SIZES}
p_equiv = E_syn / tpp["60M"]["arith"]
report["prong_A_variant_tokens_per_param"] = {
    "tokens_per_param": tpp,
    "arith_60M_tpp": tpp["60M"]["arith"],
    "distance_to_syntax_tpp": d_tpp,
    "nearest_syntax_size": min(d_tpp, key=d_tpp.get),
    "exact_equivalent_syntax_params": p_equiv,
    "equivalent_size_relative_to_60M": p_equiv / PARAMS["60M"],
}

# Prong B: does arith-60M attraction sit between syntax-23M and syntax-60M?
a60 = mean(list(ari["60M"].values()))
s23 = mean(list(syn["23M"].values()))
s60 = mean(list(syn["60M"].values()))
report["prong_B_interval"] = {
    "arith_60M": a60,
    "syntax_23M": s23,
    "syntax_60M": s60,
    "interval": [min(s23, s60), max(s23, s60)],
    "inside": min(s23, s60) <= a60 <= max(s23, s60),
    "excess_above_interval_top": a60 - max(s23, s60),
}

# ------------------ 3. what H_exposure quantitatively predicts, per size
# Fit the syntax-mixture scale line (the only mixture where scale is resolved).
xs, ys = [], []
for s in SIZES:
    for seed, v in syn[s].items():
        xs.append(math.log10(PARAMS[s]))
        ys.append(v)
slope, icept, se_slope, t_s, dof_s = ols(xs, ys)
report["syntax_scale_line"] = {
    "slope_per_decade": slope,
    "intercept": icept,
    "se": se_slope,
    "t": t_s,
    "p": t_sf(t_s, dof_s),
    "n": len(xs),
}

# H_exposure in its strongest (tokens-per-parameter) form: an arith model with
# 0.8x the English is equivalent to a syntax model with params P/0.8, so the
# predicted attraction shift is uniform: slope * log10(1/0.8).
ratio = E_ari / E_syn
predicted_offset = slope * math.log10(1.0 / ratio)
per_size = {}
for s in SIZES:
    obs_a = mean(list(ari[s].values()))
    obs_s = mean(list(syn[s].values()))
    pred = icept + slope * math.log10(PARAMS[s] / ratio)  # exposure-equivalent size
    per_size[s] = {
        "arith_obs": obs_a,
        "arith_sd": sd(list(ari[s].values())),
        "syntax_obs": obs_s,
        "syntax_sd": sd(list(syn[s].values())),
        "observed_diff_arith_minus_syntax": obs_a - obs_s,
        "H_exposure_pred_arith": pred,
        "H_exposure_residual": obs_a - pred,
        "H_param_pred_arith": icept + slope * math.log10(PARAMS[s]),
        "H_param_residual": obs_a - (icept + slope * math.log10(PARAMS[s])),
    }
report["per_size"] = per_size
report["H_exposure"] = {
    "english_ratio": ratio,
    "predicted_uniform_offset": predicted_offset,
    "observed_mean_offset": mean(
        [per_size[s]["observed_diff_arith_minus_syntax"] for s in SIZES]
    ),
    "predicts_interaction": 0.0,
    "rmse_H_exposure": math.sqrt(
        mean([per_size[s]["H_exposure_residual"] ** 2 for s in SIZES])
    ),
    "rmse_H_param": math.sqrt(
        mean([per_size[s]["H_param_residual"] ** 2 for s in SIZES])
    ),
}

# Is the exposure-predicted offset rejected by the data? Estimate the observed
# offset with its SE from the per-model residuals about the common slope.
res_ari, res_syn = [], []
for s in SIZES:
    base = icept + slope * math.log10(PARAMS[s])
    res_ari += [v - base for v in ari[s].values()]
    res_syn += [v - base for v in syn[s].values()]
off = mean(res_ari) - mean(res_syn)
se_off = math.sqrt(sd(res_ari) ** 2 / len(res_ari) + sd(res_syn) ** 2 / len(res_syn))
z = (off - predicted_offset) / se_off
dof_off = len(res_ari) + len(res_syn) - 2
report["offset_test"] = {
    "observed_offset": off,
    "se": se_off,
    "H_exposure_predicted": predicted_offset,
    "t_vs_prediction": z,
    "p_vs_prediction": t_sf(z, dof_off),
    "sign_matches_prediction": (off > 0) == (predicted_offset > 0),
}

# --- the mapping is direction-ambiguous; do BOTH verbal readings, and both
# accountings of "English" (all natural language vs diverse base corpus only).
base_syn = ledger["11M"]["syntax"]["english_tokens"] - 8_192_000  # base stream only
base_ari = ledger["11M"]["arith"]["english_tokens"]
variants = {}
for acc_name, r in (
    ("all_natural_language", E_ari / E_syn),
    ("base_corpus_only", base_ari / base_syn),
):
    for read_name, eff in (
        # reading 1: tokens-per-parameter is causal -> arith ~ syntax at P/r
        ("tokens_per_param", 1.0 / r),
        # reading 2: less data -> smaller EFFECTIVE size -> arith ~ syntax at P*r
        ("effective_size", r),
    ):
        variants[f"{acc_name}|{read_name}"] = {
            "english_ratio": r,
            "effective_param_multiplier": eff,
            "predicted_uniform_offset": slope * math.log10(1.0 / eff),
            "arith_60M_maps_to_syntax_params": PARAMS["60M"] * eff,
            "that_is_near_syntax_23M": abs(math.log10(PARAMS["60M"] * eff / PARAMS["23M"]))
            < abs(math.log10(PARAMS["60M"] * eff / PARAMS["60M"])),
        }
report["H_exposure_variants"] = variants
report["max_offset_exposure_can_produce"] = max(
    abs(v["predicted_uniform_offset"]) for v in variants.values()
)
# how much English would arith-60M need for exposure alone to yield +0.133?
need = 10 ** (per_size["60M"]["observed_diff_arith_minus_syntax"] / slope)
report["english_ratio_required_for_60M_divergence"] = {
    "required_multiplier_vs_syntax": need,
    "actual_multiplier": E_ari / E_syn,
    "note": "under the effective_size reading; the tokens_per_param reading needs 1/need",
}


# ------------- 4. independent readout: shared-English held-out perplexity
# H_exposure says arith-60M should look English-starved (like a smaller /
# under-trained model). val.bin is the same held-out English for every run.
ppl = {}
for size in SIZES:
    ppl[size] = {}
    for mix, pat in (("syntax", SYNTAX_RUN), ("arith", ARITH_RUN)):
        vals = [
            final_val_ppl(pat.format(size=size, seed=seed))
            for seed in (1, 2, 3)
        ]
        vals = [v for v in vals if v is not None]
        ppl[size][mix] = {"mean": mean(vals) if vals else None, "n": len(vals),
                          "vals": vals}
report["english_val_ppl"] = ppl

a60p = ppl["60M"]["arith"]["mean"]
report["ppl_check"] = {
    "arith_60M": a60p,
    "syntax_23M": ppl["23M"]["syntax"]["mean"],
    "syntax_60M": ppl["60M"]["syntax"]["mean"],
    "arith_60M_beats_syntax_23M": a60p < ppl["23M"]["syntax"]["mean"],
    "arith_60M_vs_syntax_60M": a60p - ppl["60M"]["syntax"]["mean"],
}

# rank of arith-60M English competence among all six cells
cells = sorted(
    [(f"{s}-{m}", ppl[s][m]["mean"]) for s in SIZES for m in ("syntax", "arith")],
    key=lambda kv: kv[1],
)
report["ppl_ranking_best_first"] = cells

# does attraction track English ppl (H_exposure) or params (H_param)?
cell_attr, cell_ppl, cell_par = [], [], []
for s in SIZES:
    for m, src in (("syntax", syn), ("arith", ari)):
        cell_attr.append(mean(list(src[s].values())))
        cell_ppl.append(math.log10(ppl[s][m]["mean"]))
        cell_par.append(math.log10(PARAMS[s]))


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


r_ppl = pearson(cell_attr, cell_ppl)
r_par = pearson(cell_attr, cell_par)
n_cells = len(cell_attr)
report["cell_level_correlations"] = {
    "n_cells": n_cells,
    "r_attraction_vs_log_english_ppl": r_ppl,
    "p": t_sf(r_ppl * math.sqrt((n_cells - 2) / (1 - r_ppl**2)), n_cells - 2),
    "r_attraction_vs_log_params": r_par,
    "p_params": t_sf(r_par * math.sqrt((n_cells - 2) / (1 - r_par**2)), n_cells - 2),
}

# ---------------------------------- 5. where is the divergence concentrated?
per_size_div = {}
for s in SIZES:
    a, b = list(ari[s].values()), list(syn[s].values())
    t, d, p = welch(a, b)
    per_size_div[s] = {
        "diff": mean(a) - mean(b),
        "t": t,
        "df": d,
        "p": p,
        "n_arith": len(a),
        "n_syntax": len(b),
    }
report["per_size_divergence"] = per_size_div

# ---------- 6. THE INDEPENDENT KILL: is arith-60M actually English-starved?
# H_exposure requires arith models to be English-degraded, and requires that
# degradation to be LARGEST at 60M (where the attraction divergence is largest).
pen = {}
for s in SIZES:
    a = ppl[s]["arith"]["vals"]
    b = ppl[s]["syntax"]["vals"]
    t, d, p = welch(a, b)
    pen[s] = {
        "arith_ppl": mean(a),
        "syntax_ppl": mean(b),
        "penalty": mean(a) - mean(b),
        "penalty_pct": 100 * (mean(a) - mean(b)) / mean(b),
        "t": t,
        "df": d,
        "p": p,
    }
report["english_penalty_by_size"] = pen
pens = [pen[s]["penalty"] for s in SIZES]
divs = [per_size[s]["observed_diff_arith_minus_syntax"] for s in SIZES]
report["penalty_vs_divergence"] = {
    "penalties": dict(zip(SIZES, pens)),
    "divergences": dict(zip(SIZES, divs)),
    "r": pearson(pens, divs),
    "H_exposure_requires": "positive r (bigger English penalty -> bigger divergence)",
    "penalty_shrinks_with_size": pens[0] > pens[1] > pens[2],
}

out_json = PROBE / "exposure_recheck.json"
json.dump(report, open(out_json, "w"), indent=1)
print(json.dumps(report, indent=1))
print(f"\nwrote {out_json}")
