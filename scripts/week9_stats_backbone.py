"""Week 9 — the paper's formal statistics, computed once, mechanically.

WHY: every headline number in the write-up must have exactly ONE owner that
computes it, so the abstract, the results section and the discussion cannot
drift into quoting three versions of the same figure. This script is the owner
for the arithmetic Detect/Reopen cluster and the Week-8 circuit cluster. It
does NOT recompute the syntax-null or SVA clusters — those already have owners
(`szsweep_gate.py`, `agreement_60M_replication_readout.py`,
`agreement_slope_imbalance_check.py`); the backbone note maps number -> owner.

Inputs are the per-seed endpoints already published in the run read-outs (cited
per block below). Nothing here re-reads checkpoints; it is pure statistics over
recorded endpoints, so it is cheap to re-run and safe to re-run.

Design decisions encoded here (rationale in notes/week9-stats-backbone.md):
  * Seeds are CROSSED across conditions (same seed = same init), so the
    design-appropriate Detect statistic is the PER-SEED PAIRED deficit, not an
    unpaired group comparison. Welch is reported beside it for continuity with
    the pre-registered gate, which used Welch.
  * The pre-registered `accuracy ~ delay x model_size` mixed-effects model is
    NOT estimable on the surviving design (see note §2.1) and is not attempted.
  * Any non-significant moderation is reported WITH its MDE, so "no size effect"
    can never be read as "size effects are excluded".
  * n=1 intervention arms are scored against the NULL'S CROSS-SEED RANGE, not
    against its mean — the only honest use of a single-seed arm when the null
    is itself seed-variable.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "logs" / "week9_stats"

# ---------------------------------------------------------------- input data
# Arithmetic exact-match @ step 5000. Seeds 1/2/3 in order.
# 23M   : logs/week4_close/results.md (Items 2,3), logs/week4_arith_pilot/
#         seeds23_replication_results.md
# 60M/11M: logs/week5-7_arith_scale/followups_readout.md (A,B,D)
ARITH = {
    "11M": {  # pre-registered cell STOPped at Rule-1 FLOOR (anchor < 0.60)
        "d0": [0.556, 0.538, 0.520],
        "d50": [0.304, 0.332, 0.436],
        "lr_shift": [0.726, 0.768, 0.746],
        "d0_lr_shift": [0.612, 0.704, 0.644],  # H2/H3 discriminator
    },
    "23M": {
        "d0": [0.798, 0.792, 0.784],
        "d50": [0.524, 0.626, 0.582],  # seed1 = hardened-code canonical 0.524
        "lr_shift": [0.752, 0.778, 0.772],
        "cram_lr_shift": [0.746, 0.794, 0.804],
    },
    "60M": {
        "d0": [0.798, 0.840, 0.848],
        "d50": [0.574, 0.646, 0.580],
        "lr_shift": [0.796, 0.818, 0.774],
    },
}

# 23M single-seed (seed 1) intervention arms, all forked from the identical
# common checkpoint. logs/week4_arith_pilot/report.md
ARMS_N1_23M = {
    "reset_embedding": 0.586,
    "reset_plus_spike": 0.576,
    "lr_spike": 0.536,
    "cram": 0.574,
    "freeze": 0.332,
}

# 60M forgetting gate, per seed: c-b = lr_shift val_ppl - d50 null val_ppl,
# and the matched-LR lead (null read at its step-2500, same LR phase).
# logs/week5-7_arith_scale/followups_readout.md (B,C)
FORGET_60M = {"breach_c_minus_b": [1.790, 1.601, 1.619],
              "matched_lr_lead": [2.20, 2.63, 2.60],
              "gate": 1.0}

# Week-8 circuit identity is loaded from its OWNER artifact rather than
# retyped, so the group means here cannot drift from the ones in
# logs/week8_circuit_identity/results.md.
#
# PRIMARY is the *_cleaneval artifact (2026-07-28): the original run built its
# eval set with ArithConfig(seed=<run seed>), and only seed 1's draw is held out
# of training — seeds 2/3 were scored on draws containing ~17-19% trained-on
# problems, at BOTH sizes. The superseded artifact is still read, purely to
# report what the correction moved.
W8 = ROOT / "logs" / "week8_circuit_identity"
SCALE_VS_DEFICIT = W8 / "scale_vs_deficit_cleaneval.json"
SCALE_VS_DEFICIT_SUPERSEDED = W8 / "scale_vs_deficit.json"

# Main-harness endpoints for the six DIAGNOSED triads, keyed the same way, for
# the cross-harness agreement check. NOTE 23M-s1 B = 0.506: the diagnosed
# delayed model is the ORIGINAL seed-1 run, not the from-scratch `_recheck`
# re-train whose 0.524 is the canonical Detect null. Same run_id ->
# same weights; different run_id -> different model. Do not swap them.
MAIN_HARNESS_TRIAD_ACC = {
    ("23M", 1): {"A": 0.798, "B": 0.506, "C": 0.752},
    ("23M", 2): {"A": 0.792, "B": 0.626, "C": 0.778},
    ("23M", 3): {"A": 0.784, "B": 0.582, "C": 0.772},
    ("60M", 1): {"A": 0.798, "B": 0.574, "C": 0.796},
    ("60M", 2): {"A": 0.840, "B": 0.646, "C": 0.818},
    ("60M", 3): {"A": 0.848, "B": 0.580, "C": 0.774},
}
EVAL_N = 500  # both harnesses score 500 held-out problems


# ------------------------------------------------------------------- helpers
def desc(x) -> dict:
    a = np.asarray(x, float)
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "values": [float(v) for v in a]}


def hedges_g(a, b) -> float:
    """Paired-design effect size on the differences (g_av style, small-n
    corrected). Reported because n=3 t-values are unstable but the
    standardised effect is what generalises to a replication attempt."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    g = d.mean() / d.std(ddof=1)
    return float(g * (1 - 3 / (4 * n - 5)))  # small-sample correction


def paired_test(treat, ctrl, alternative="greater") -> dict:
    t, p = stats.ttest_rel(treat, ctrl, alternative=alternative)
    d = np.asarray(treat, float) - np.asarray(ctrl, float)
    return {"per_seed_diff": [float(v) for v in d], "mean_diff": float(d.mean()),
            "sd_diff": float(d.std(ddof=1)), "t": float(t), "df": len(d) - 1,
            "p": float(p), "hedges_g": hedges_g(treat, ctrl),
            "signs_agree": bool(np.all(d > 0) or np.all(d < 0))}


def welch(a, b, alternative="greater") -> dict:
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    return {"t": float(t), "p": float(p)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values within a declared family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = float(min(1.0, running))
    return adj


def mde_two_sample(sd: float, n: int, alpha=0.05, power=0.8, sides=2) -> float:
    """Smallest difference this design could have detected. Solved on the
    noncentral t, not the normal approximation — at n=3 the difference is large
    and the normal approximation would understate the MDE."""
    df = 2 * (n - 1)
    crit = stats.t.ppf(1 - alpha / sides, df)
    lo, hi = 1e-6, 50.0 * max(sd, 1e-9)
    for _ in range(200):
        mid = (lo + hi) / 2
        ncp = mid / (sd * math.sqrt(2.0 / n))
        got = 1 - stats.nct.cdf(crit, df, ncp)
        if got < power:
            lo = mid
        else:
            hi = mid
    return float(hi)


def spearman_exact_p(x, y, alternative="less") -> tuple[float, float]:
    """Exact permutation p for Spearman rho. At n=6 there are only 720
    permutations, so the asymptotic/t-approximation p (which is what
    scipy reports) is replaceable by the exact one — worth doing when the
    whole claim rests on six points."""
    rho = float(stats.spearmanr(x, y).statistic)
    ry = stats.rankdata(y)
    rx = stats.rankdata(x)
    count = 0
    perms = list(itertools.permutations(ry))
    for p in perms:
        r = float(stats.spearmanr(rx, p).statistic)
        if (r <= rho + 1e-12) if alternative == "less" else (r >= rho - 1e-12):
            count += 1
    return rho, count / len(perms)


# ------------------------------------------------------------------ clusters
def cluster_detect() -> dict:
    """C1 — does the arithmetic critical period exist, and does it scale?"""
    out = {"per_size": {}, "notes": {}}
    raw_p = {}
    for size in ("11M", "23M", "60M"):
        d0, d50 = ARITH[size]["d0"], ARITH[size]["d50"]
        pt = paired_test(d0, d50, "greater")
        out["per_size"][size] = {
            "early_anchor": desc(d0), "delayed_null": desc(d50),
            "paired_deficit": pt, "welch": welch(d0, d50, "greater"),
            "floor_rule_anchor_ge_0.60": bool(np.mean(d0) >= 0.60),
        }
        if size != "11M":  # 11M excluded by the pre-registered FLOOR rule
            raw_p[size] = pt["p"]

    out["family_F1_holm"] = holm(raw_p)
    out["notes"]["11M"] = ("EXCLUDED from the confirmatory family by the "
                           "pre-registered Rule-1 FLOOR (anchor 0.538 < 0.60): "
                           "a capacity/budget boundary, NOT a 'no critical "
                           "period' timing claim.")

    # Size moderation: is the deficit itself bigger at 60M than 23M?
    d23 = np.array(ARITH["23M"]["d0"]) - np.array(ARITH["23M"]["d50"])
    d60 = np.array(ARITH["60M"]["d0"]) - np.array(ARITH["60M"]["d50"])
    pooled_sd = math.sqrt((d23.var(ddof=1) + d60.var(ddof=1)) / 2)
    out["size_moderation"] = {
        "deficit_23M": desc(d23), "deficit_60M": desc(d60),
        "welch_two_sided": {"t": float(stats.ttest_ind(d60, d23,
                                                       equal_var=False).statistic),
                            "p": float(stats.ttest_ind(d60, d23,
                                                       equal_var=False).pvalue)},
        "pooled_sd": float(pooled_sd),
        "mde_80pct_power": mde_two_sample(pooled_sd, 3),
        "reading": ("Non-significant moderation. Report WITH the MDE: this "
                    "design can only exclude size effects LARGER than the MDE, "
                    "which is itself a large fraction of the deficit. "
                    "'Scale-robust' means 'the CP is present and of similar "
                    "magnitude at both sizes', never 'size provably does not "
                    "matter'."),
    }
    return out


def cluster_reopen() -> dict:
    """C2 — does lr_shift reopen, and at what general-LM cost?"""
    out = {"per_size": {}}
    raw_p = {}
    for size in ("11M", "23M", "60M"):
        d0, d50 = ARITH[size]["d0"], ARITH[size]["d50"]
        arm = ARITH[size]["lr_shift"]
        gate = 0.90 * float(np.mean(d0))
        lift = paired_test(arm, d50, "greater")
        out["per_size"][size] = {
            "treated_endpoint": desc(arm),          # both numbers, never a ratio
            "delayed_null_endpoint": desc(d50),
            "early_anchor_endpoint": desc(d0),
            "gate_0.90x_own_d0": gate,
            "gate_margin": float(np.mean(arm) - gate),
            "gate_pass": bool(np.mean(arm) >= gate),
            "per_seed_lift_over_own_null": lift,
            "sd_collapse_null_to_treated": {
                "null_sd": desc(d50)["sd"], "treated_sd": desc(arm)["sd"]},
        }
        if size != "11M":
            raw_p[size] = lift["p"]
    out["family_F2_holm"] = holm(raw_p)

    # Same-endpoint band (pre-registered, memory/week5-7-arith-scale-check).
    out["same_endpoint_band"] = {
        "band": [0.722, 0.812],
        "23M": float(np.mean(ARITH["23M"]["lr_shift"])),
        "60M": float(np.mean(ARITH["60M"]["lr_shift"])),
        "both_in_band": bool(all(0.722 <= np.mean(ARITH[s]["lr_shift"]) <= 0.812
                                 for s in ("23M", "60M"))),
    }

    # n=1 arms scored against the null's cross-seed RANGE.
    null = ARITH["23M"]["d50"]
    lo, hi = float(min(null)), float(max(null))
    out["single_seed_arms_23M"] = {
        "null_cross_seed_range": [lo, hi],
        "rule": ("An n=1 arm inside the null's own cross-seed range cannot be "
                 "distinguished from the null and is scored NO DETECTABLE "
                 "RECOVERY (n=1), not 'fails'. Only an arm outside that range "
                 "supports a directional claim at n=1."),
        "arms": {k: {"endpoint": v,
                     "inside_null_range": bool(lo <= v <= hi),
                     "verdict": ("no detectable recovery (n=1)" if lo <= v <= hi
                                 else ("harmful (below entire null range, n=1)"
                                       if v < lo else "recovery (n=1)"))}
                 for k, v in ARMS_N1_23M.items()},
    }

    # cram+lr_shift vs lr_shift: sign-disagreeing near-zero, reported per seed.
    out["cram_plus_lr_shift_vs_lr_shift_23M"] = paired_test(
        ARITH["23M"]["cram_lr_shift"], ARITH["23M"]["lr_shift"], "two-sided")

    # 11M mechanism decomposition (exploratory; cell stays STOPped).
    lr_delayed = float(np.mean(ARITH["11M"]["lr_shift"])
                       - np.mean(ARITH["11M"]["d50"]))
    lr_early = float(np.mean(ARITH["11M"]["d0_lr_shift"])
                     - np.mean(ARITH["11M"]["d0"]))
    out["mechanism_11M_H2_H3"] = {
        "delayed_lift": lr_delayed, "generic_component_H2": lr_early,
        "alignment_component_H3": lr_delayed - lr_early,
        "pct_alignment": 100 * (lr_delayed - lr_early) / lr_delayed,
        "label": "EXPLORATORY — pre-registered 11M cell remains FLOOR/STOPped.",
    }

    # Forgetting: the 3-number decomposition, never a single ratio.
    breach = FORGET_60M["breach_c_minus_b"]
    lead = FORGET_60M["matched_lr_lead"]
    out["forgetting_60M"] = {
        "breach_c_minus_b": desc(breach), "gate": FORGET_60M["gate"],
        "gate_tripped_all_seeds": bool(all(b > FORGET_60M["gate"] for b in breach)),
        "matched_lr_lead": desc(lead),
        "lead_exceeds_breach_all_seeds": bool(all(l > b for l, b in zip(lead, breach))),
        "upper_bound_on_residual_cost_ppl": float(np.mean(breach)),
        "label": "BOUNDED-NOT-EXCLUDED",
        "reading": ("Gate honored as written: it trips at the fixed 5000-step "
                    "horizon in all 3 seeds. The arm is read mid-anneal at 6x "
                    "the null's terminal LR and leads the null by 2.48+/-0.24 "
                    "ppl at matched LR — a lead exceeding the breach. Residual "
                    "cost is bounded ABOVE by the breach; the anneal slope is "
                    "unmeasured, so no point estimate is quoted."),
    }
    return out


def cross_harness_agreement(clean_rows, superseded_rows) -> dict:
    """How far apart do the two eval harnesses put the SAME checkpoint?

    After the fix both harnesses score the same held-out 500 problems, so the
    residual gap is pure inference-level noise and gives the paper its true
    measurement floor. The superseded rows are scored against the same main-
    harness numbers to show what the wrong draw was contributing — which is the
    quantity that decides whether any past claim moved.
    """
    def gaps(rows):
        out = []
        for r in rows:
            main = MAIN_HARNESS_TRIAD_ACC[(r["size"], r["seed"])]
            for arm in ("A", "B", "C"):
                out.append({"triad": f"{r['size']}-s{r['seed']}", "arm": arm,
                            "diagnosis": r[arm], "main": main[arm],
                            "abs_delta": round(abs(r[arm] - main[arm]), 4)})
        return out

    clean, sup = gaps(clean_rows), gaps(superseded_rows)
    cd = np.array([g["abs_delta"] for g in clean])
    sd_ = np.array([g["abs_delta"] for g in sup])
    # seed-1 triads were already on the held-out draw in the superseded run
    sup_wrong = np.array([g["abs_delta"] for g, r in
                          zip([g for g in sup], sum([[r] * 3 for r in superseded_rows], []))
                          if r["seed"] != 1])
    p = 0.6
    return {
        "per_model_clean": clean,
        "clean_all_triads": {"n": int(cd.size), "mean_abs_delta": float(cd.mean()),
                             "max_abs_delta": float(cd.max())},
        "superseded_all_triads": {"n": int(sd_.size), "mean_abs_delta": float(sd_.mean()),
                                  "max_abs_delta": float(sd_.max())},
        "superseded_contaminated_triads_only": {
            "n": int(sup_wrong.size), "mean_abs_delta": float(sup_wrong.mean()),
            "max_abs_delta": float(sup_wrong.max())},
        "expected_sd_of_two_draw_difference": float(math.sqrt(2 * p * (1 - p) / EVAL_N)),
        "reading": ("On the corrected draw the two harnesses agree to ~0.005 on "
                    "every model — that, not 0.03, is the measurement floor for "
                    "identical items. The superseded spread was the contaminated "
                    "draw, not harness disagreement."),
    }


def cluster_circuit() -> dict:
    """C4 — repair vs reorganization. Single-method (mean-ablation) by
    demonstration: CKA, single-head patching and pruning were each shown
    NON-DISCRIMINATING on this exact contrast."""
    rows = json.loads(SCALE_VS_DEFICIT.read_text(encoding="utf-8"))["rows"]
    names = [f"{r['size']}-s{r['seed']}" for r in rows]
    deficit = np.array([r["deficit"] for r in rows])
    transfer = np.array([r["transfer"] for r in rows])

    rho, p_exact = spearman_exact_p(deficit, transfer, "less")
    pear = stats.pearsonr(deficit, transfer)

    loo = {}
    for i, nm in enumerate(names):
        m = np.ones(len(names), bool)
        m[i] = False
        r, p = spearman_exact_p(deficit[m], transfer[m], "less")
        loo[f"drop_{nm}"] = {"rho": r, "p_exact": p, "n": int(m.sum())}

    d23 = np.array([r["transfer"] for r in rows if r["size"] == "23M"])
    d60 = np.array([r["transfer"] for r in rows if r["size"] == "60M"])
    # What the contamination correction moved, point by point.
    old_rows = json.loads(SCALE_VS_DEFICIT_SUPERSEDED.read_text(encoding="utf-8"))["rows"]
    old = {(r["size"], r["seed"]): r for r in old_rows}
    delta = [{"triad": f"{r['size']}-s{r['seed']}",
              "deficit_superseded": round(old[(r["size"], r["seed"])]["deficit"], 4),
              "deficit_clean": round(r["deficit"], 4),
              "transfer_superseded": old[(r["size"], r["seed"])]["transfer"],
              "transfer_clean": r["transfer"]} for r in rows]
    o_def = np.array([d["deficit_superseded"] for d in delta])
    o_tr = np.array([d["transfer_superseded"] for d in delta])
    o_rho, o_p = spearman_exact_p(o_def, o_tr, "less")

    return {
        "triads": [{"triad": n, "deficit": float(x), "transfer": float(y)}
                   for n, x, y in zip(names, deficit, transfer)],
        "contamination_correction": {
            "per_triad": delta,
            "superseded_pooled_spearman": {"rho": o_rho, "p_exact": o_p},
            "note": ("Seed-1 triads are unchanged at both sizes (their draw was "
                     "already the held-out one) — that is the no-op control "
                     "showing the eval draw is the only thing that moved."),
        },
        "cross_harness_agreement": cross_harness_agreement(rows, old_rows),
        "pooled_deficit_vs_transfer": {
            "n": len(names), "spearman_rho": rho,
            "p_exact_permutation": p_exact,
            "p_scipy_asymptotic": float(stats.spearmanr(deficit, transfer).pvalue),
            "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
            "note": ("Exact permutation p replaces the asymptotic p: at n=6 "
                     "there are only 720 orderings, so the exact value is "
                     "computable and the approximation is unnecessary."),
        },
        "leave_one_out_spearman": loo,
        "loo_robust": bool(all(v["p_exact"] < 0.05 for v in loo.values())),
        "by_size": {
            "23M": desc(d23), "60M": desc(d60),
            "welch_two_sided_p": float(stats.ttest_ind(d60, d23,
                                                       equal_var=False).pvalue),
            "variance_ratio_23M_over_60M": float(d23.std(ddof=1) / d60.std(ddof=1)),
            "pooled_sd": float(math.sqrt((d23.var(ddof=1) + d60.var(ddof=1)) / 2)),
            "mde_80pct_power": mde_two_sample(
                math.sqrt((d23.var(ddof=1) + d60.var(ddof=1)) / 2), 3),
            "outcome_span_all_triads": float(
                max(list(d23) + list(d60)) - min(list(d23) + list(d60))),
            "reading": ("UNDERPOWERED, NOT NULL. The contamination correction made "
                        "the deficit ranges overlap, so size is no longer collinear "
                        "with deficit depth and the comparison is ESTIMABLE. It is "
                        "not thereby ANSWERED: MDE at 80% power is 0.809, which "
                        "EXCEEDS the 0.681 span of the transfer ratio across all six "
                        "triads, so no realizable difference could have been detected "
                        "and the p=0.39 excludes nothing. Contrast C1, whose MDE "
                        "(0.144) sits below its deficit (0.21) and so constrains. "
                        "Report as untested; never as a tested null. The one "
                        "directional size effect is REPRODUCIBILITY: SD 0.349 (23M) "
                        "vs 0.131 (60M), 2.7x — not inferential at n=3/group."),
        },
        "method_scope": {
            "discriminating_methods": ["mean_ablation_reciprocal_transfer"],
            "non_discriminating_verified": {
                "CKA": "0.66-0.82 for every triad regardless of verdict",
                "single_head_patching": "r=0.045 (p=0.93) vs ablation, n=6",
                "pruning": "sign of the answer flips with lambda; feasibility NO-GO",
                "whole_layer_ablation": "saturates (any core layer -> acc~0)",
            },
            "label": "SINGLE-METHOD (by demonstration, not by omission)",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {"detect": cluster_detect(), "reopen": cluster_reopen(),
           "circuit": cluster_circuit()}
    (OUT_DIR / "backbone_stats.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    d, r, c = res["detect"], res["reopen"], res["circuit"]
    L = []
    L.append("# Week 9 — computed statistical backbone (arith + circuit clusters)\n")
    L.append("Owner: `scripts/week9_stats_backbone.py`. Re-runnable, pure stats "
             "over recorded per-seed endpoints.\n")

    L.append("\n## C1 Detect — per-seed paired deficit (primary)\n")
    L.append("| size | early anchor | delayed null | deficit mean±SD | paired t (df2) | p | Holm | g |")
    L.append("|:--|:--|:--|:--|--:|--:|--:|--:|")
    for s in ("11M", "23M", "60M"):
        b = d["per_size"][s]
        pt = b["paired_deficit"]
        hp = d["family_F1_holm"].get(s)
        L.append(f"| {s} | {b['early_anchor']['mean']:.3f} ± {b['early_anchor']['sd']:.3f} "
                 f"| {b['delayed_null']['mean']:.3f} ± {b['delayed_null']['sd']:.3f} "
                 f"| {pt['mean_diff']:.3f} ± {pt['sd_diff']:.3f} | {pt['t']:.2f} "
                 f"| {pt['p']:.4f} | {'—' if hp is None else f'{hp:.4f}'} "
                 f"| {pt['hedges_g']:.2f} |")
    sm = d["size_moderation"]
    L.append(f"\nSize moderation (60M vs 23M deficit): p = {sm['welch_two_sided']['p']:.2f}, "
             f"**MDE at 80% power = {sm['mde_80pct_power']:.3f}** — i.e. only size effects "
             f"larger than {sm['mde_80pct_power']:.2f} are excluded, against a deficit of "
             f"~{sm['deficit_23M']['mean']:.2f}.\n")

    L.append("\n## C2 Reopen — endpoints, never collapsed into a ratio\n")
    L.append("| size | treated (lr_shift) | delayed null | early anchor | gate 0.90×d0 | margin | pass | lift p |")
    L.append("|:--|:--|:--|:--|--:|--:|:--:|--:|")
    for s in ("11M", "23M", "60M"):
        b = r["per_size"][s]
        L.append(f"| {s} | {b['treated_endpoint']['mean']:.3f} ± {b['treated_endpoint']['sd']:.3f} "
                 f"| {b['delayed_null_endpoint']['mean']:.3f} ± {b['delayed_null_endpoint']['sd']:.3f} "
                 f"| {b['early_anchor_endpoint']['mean']:.3f} ± {b['early_anchor_endpoint']['sd']:.3f} "
                 f"| {b['gate_0.90x_own_d0']:.3f} | {b['gate_margin']:+.3f} "
                 f"| {'PASS' if b['gate_pass'] else 'FAIL'} "
                 f"| {b['per_seed_lift_over_own_null']['p']:.4f} |")
    L.append("\n### n=1 arms vs the null's cross-seed range "
             f"{r['single_seed_arms_23M']['null_cross_seed_range']}\n")
    L.append("| arm | endpoint | verdict |")
    L.append("|:--|--:|:--|")
    for k, v in r["single_seed_arms_23M"]["arms"].items():
        L.append(f"| {k} | {v['endpoint']:.3f} | {v['verdict']} |")
    f = r["forgetting_60M"]
    L.append(f"\n60M forgetting: breach {f['breach_c_minus_b']['mean']:+.3f} ± "
             f"{f['breach_c_minus_b']['sd']:.3f} ppl (gate {f['gate']:+.1f}, tripped all seeds) "
             f"vs matched-LR lead {f['matched_lr_lead']['mean']:+.3f} ± "
             f"{f['matched_lr_lead']['sd']:.3f} → **{f['label']}**, residual bounded "
             f"≤ {f['upper_bound_on_residual_cost_ppl']:.2f} ppl.\n")

    p = c["pooled_deficit_vs_transfer"]
    L.append("\n## C4 Circuit identity — deficit depth vs ablation transfer\n")
    L.append(f"Spearman ρ = {p['spearman_rho']:.3f}, **exact permutation "
             f"p = {p['p_exact_permutation']:.4f}** (n = {p['n']}; asymptotic "
             f"p = {p['p_scipy_asymptotic']:.4f}).\n")
    L.append("| dropped point | ρ | exact p | n |")
    L.append("|:--|--:|--:|--:|")
    for k, v in c["leave_one_out_spearman"].items():
        L.append(f"| {k.replace('drop_', '')} | {v['rho']:.3f} | {v['p_exact']:.4f} | {v['n']} |")
    n_fragile = sum(1 for v in c["leave_one_out_spearman"].values()
                    if v["p_exact"] >= 0.05)
    L.append(f"\nLeave-one-out robust at α=0.05: **{c['loo_robust']}** — "
             + ("the relation does not rest on any single triad.\n" if c["loo_robust"]
                else f"{n_fragile} of {len(c['leave_one_out_spearman'])} single-point "
                     "deletions push it above 0.05, so it is carried by the whole "
                     "six-point set rather than being robust within it.\n"))
    cc = c["contamination_correction"]
    L.append("\n### Contamination correction (2026-07-28)\n")
    L.append("| triad | deficit superseded → clean | transfer superseded → clean |")
    L.append("|:--|:--|:--|")
    for d in cc["per_triad"]:
        L.append(f"| {d['triad']} | {d['deficit_superseded']:.3f} → "
                 f"{d['deficit_clean']:.3f} | {d['transfer_superseded']:.2f} → "
                 f"{d['transfer_clean']:.2f} |")
    L.append(f"\nSuperseded pooled Spearman was ρ = "
             f"{cc['superseded_pooled_spearman']['rho']:.3f} "
             f"(exact p = {cc['superseded_pooled_spearman']['p_exact']:.4f}); "
             "seed-1 triads are unchanged at both sizes (no-op control).\n")
    ch = c["cross_harness_agreement"]
    L.append("\n### Cross-harness agreement (same checkpoints, two eval harnesses)\n")
    L.append(f"- corrected draw, all 6 triads: mean |Δ| "
             f"{ch['clean_all_triads']['mean_abs_delta']:.4f}, "
             f"max {ch['clean_all_triads']['max_abs_delta']:.4f} "
             f"(n={ch['clean_all_triads']['n']})")
    L.append(f"- superseded, contaminated triads only: mean |Δ| "
             f"{ch['superseded_contaminated_triads_only']['mean_abs_delta']:.4f}, "
             f"max {ch['superseded_contaminated_triads_only']['max_abs_delta']:.4f} "
             f"(n={ch['superseded_contaminated_triads_only']['n']})")
    L.append(f"- binomial prediction for two independent {EVAL_N}-item draws: SD "
             f"{ch['expected_sd_of_two_draw_difference']:.4f}\n")
    L.append("**Measurement floor is ~0.005 on identical items** — not 0.03; that "
             "spread was the contaminated draw, not harness disagreement.\n")
    bs = c["by_size"]
    L.append(f"By size: 23M {bs['23M']['mean']:.3f} ± {bs['23M']['sd']:.3f} vs "
             f"60M {bs['60M']['mean']:.3f} ± {bs['60M']['sd']:.3f}, Welch p = "
             f"{bs['welch_two_sided_p']:.2f}; SD ratio {bs['variance_ratio_23M_over_60M']:.1f}×.\n")
    L.append(f"**Underpowered, not null:** MDE at 80% power = {bs['mde_80pct_power']:.3f}, "
             f"vs an outcome span of {bs['outcome_span_all_triads']:.3f} across all six "
             f"triads. The design could not have detected any realizable difference, so "
             f"p = {bs['welch_two_sided_p']:.2f} excludes nothing. Estimable (deficit no "
             f"longer collinear with size) ≠ answered.\n")

    (OUT_DIR / "backbone_stats.md").write_text("\n".join(L), encoding="utf-8")
    # stdout is cp1252 on this box; the markdown file keeps the unicode.
    print("\n".join(L).encode("ascii", "replace").decode("ascii"))
    print(f"\nwrote {OUT_DIR/'backbone_stats.json'} and backbone_stats.md")


if __name__ == "__main__":
    main()
