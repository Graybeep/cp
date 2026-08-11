#!/usr/bin/env python
"""H2/H3 discriminator at 23M — is lr_shift a generic optimizer booster?

WHY THIS RUN EXISTS
-------------------
The paper's central mechanistic claim (§4.4) is that closure at d50 is largely
a property of the optimizer's decayed LR schedule. The obvious reviewer
challenge is "then lr_shift is just a bigger learning rate, and it would lift
ANY run" — i.e. the reopening effect is generic accuracy inflation rather than
anything about the delayed model's state.

The discriminator is to apply the identical intervention to the EARLY (d0) run
and measure its lift there. That test existed only at 11M
(scripts/arith_scale_gate.py block D), which is the size the pre-registered
Rule-1 FLOOR excludes, so the paper labels it "exploratory, carries no
headline". This script repeats it at 23M, a confirmatory size, from the same
common d0 step-2500 checkpoints.

DECOMPOSITION (identical to the 11M block — both lifts over each arm's OWN null,
never a comparison of the two boosted endpoints, whose baselines differ):
    delayed_lift  = lr_shift(d50) - d50     # the reopening effect
    generic  (H2) = lr_shift(d0)  - d0      # lr_shift's lift on the EARLY run
    alignment(H3) = delayed_lift - generic  # requires the delayed profile

Reporting follows the standing rule: absolute endpoints beside their nulls,
per-seed paired lifts, never a percent-of-gap headline.

Outputs logs/week9_h2h3_23M/h2h3_result.{json,md}.
"""
import csv, json, os, statistics as st
from math import sqrt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "logs", "week9_h2h3_23M")
SEEDS = (1, 2, 3)


def metric(run_id, col, step=5000):
    f = os.path.join(ROOT, "logs", run_id, "metrics.csv")
    if not os.path.isfile(f):
        return None
    r = None
    for row in csv.DictReader(open(f)):
        try:
            if int(float(row.get("step", -1))) == step:
                r = row
        except ValueError:
            pass
    if r is None:
        return None
    for k in r:
        if k.strip() == col:
            try:
                return float(r[k])
            except (TypeError, ValueError):
                return None
    return None


def acc(rid): return metric(rid, "arith_accuracy")
def ppl(rid): return metric(rid, "val_ppl")


def msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), []
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0), xs


def paired_t_one_sided(diffs):
    """One-sided paired t (H1: mean(diffs) > 0). n=3 -> df=2."""
    n = len(diffs)
    m = st.mean(diffs)
    if n < 2:
        return float("nan"), float("nan")
    s = st.stdev(diffs)
    if s == 0:
        return float("inf"), 0.0
    t = m / (s / sqrt(n))
    # survival of Student-t with df=n-1, computed without scipy
    df = n - 1
    try:
        from scipy import stats as _s
        p = float(_s.t.sf(t, df))
    except Exception:
        p = float("nan")
    return t, p


# ------------------------------------------------------------------ conditions
d0    = [acc(f"arith_delay0_exp20_size23M_seed{s}") for s in SEEDS]

# d50 seed 1: the CANONICAL Detect null is the hardened-code recheck run
# (0.524), not the pre-hardening run this run_id logs (0.506). They are
# DIFFERENT TRAINED MODELS, not two readings of one (week9-stats-backbone.md
# rule 2, logs/week4_close/results.md:13). week9_stats_backbone.py pairs
# lr_shift against the canonical value, so this script does too — otherwise
# the H2/H3 lift would not reconcile with the paper's reported lift.
# Sensitivity to the alternative is reported below; it does not move the verdict.
D50_CANONICAL_SEED1 = 0.524
d50 = [acc(f"arith_delay50_exp20_size23M_seed{s}") for s in SEEDS]
d50_forkline = list(d50)                 # as-logged (seed 1 = 0.506)
d50[0] = D50_CANONICAL_SEED1
lrs   = [acc(f"arith_delay50_exp20_size23M_seed{s}_ivlrshift0at2501") for s in SEEDS]
disc  = [acc(f"arith_delay0_exp20_size23M_seed{s}_ivlrshift0at2501") for s in SEEDS]

d0_p   = [ppl(f"arith_delay0_exp20_size23M_seed{s}") for s in SEEDS]
disc_p = [ppl(f"arith_delay0_exp20_size23M_seed{s}_ivlrshift0at2501") for s in SEEDS]

missing = [n for n, v in [("d0", d0), ("d50", d50), ("lr_shift(d50)", lrs),
                          ("lr_shift(d0) DISCRIM", disc)]
           if any(x is None for x in v)]
if missing:
    raise SystemExit(f"missing step-5000 rows for: {missing} — runs incomplete")

m_d0, s_d0, _ = msd(d0)
m_d50, s_d50, _ = msd(d50)
m_lrs, s_lrs, _ = msd(lrs)
m_disc, s_disc, _ = msd(disc)

# per-seed paired lifts (the reportable form under the standing rule)
delayed_per_seed = [lrs[i] - d50[i] for i in range(3)]
generic_per_seed = [disc[i] - d0[i] for i in range(3)]
align_per_seed = [delayed_per_seed[i] - generic_per_seed[i] for i in range(3)]

delayed_lift = st.mean(delayed_per_seed)
generic_h2 = st.mean(generic_per_seed)
align_h3 = delayed_lift - generic_h2
h2_frac = generic_h2 / delayed_lift if delayed_lift else float("nan")
h3_frac = align_h3 / delayed_lift if delayed_lift else float("nan")

t_align, p_align = paired_t_one_sided(align_per_seed)

# Sensitivity: the same decomposition against the AS-LOGGED seed-1 null (0.506),
# i.e. the model lr_shift was literally forked from. Reported so the choice of
# null is visible rather than silent; it must not change the verdict.
_delayed_fl = [lrs[i] - d50_forkline[i] for i in range(3)]
_align_fl = [_delayed_fl[i] - generic_per_seed[i] for i in range(3)]
sens = {
    "d50_as_logged": d50_forkline,
    "delayed_lift": st.mean(_delayed_fl),
    "alignment_H3": st.mean(_align_fl),
    "H3_frac": (st.mean(_align_fl) / st.mean(_delayed_fl)
                if st.mean(_delayed_fl) else float("nan")),
}

if align_h3 > generic_h2:
    verdict = ("H3 DOMINANT (exposure-alignment). lr_shift lifts the DELAYED run "
               "%+.3f but the EARLY run only %+.3f, so %.0f%% of the reopening "
               "effect requires the delayed-exposure profile; only ~%.0f%% is a "
               "generic optimizer boost." % (delayed_lift, generic_h2,
                                             100 * h3_frac, 100 * h2_frac))
elif generic_h2 > align_h3 and generic_h2 >= 0.10:
    verdict = ("H2 DOMINANT (generic LR-booster): lr_shift lifts the early run "
               "%+.3f, comparable to or above the alignment residual %+.3f."
               % (generic_h2, align_h3))
else:
    verdict = ("MIXED: delayed lift %+.3f = generic %+.3f + alignment %+.3f "
               "(neither dominates)." % (delayed_lift, generic_h2, align_h3))

# 11M comparison (the exploratory precedent this run promotes to a
# confirmatory size) — hardcoded from logs/week5-7_arith_scale/gate_result.json
ELEVEN = {"delayed_lift": 0.389, "generic_H2": 0.115, "alignment_H3": 0.274,
          "H3_frac": 0.704}

out = {
    "size": "23M", "seeds": list(SEEDS), "step": 5000,
    "endpoints": {
        "d0": {"mean": m_d0, "sd": s_d0, "vals": d0},
        "d50": {"mean": m_d50, "sd": s_d50, "vals": d50},
        "lr_shift(d50)": {"mean": m_lrs, "sd": s_lrs, "vals": lrs},
        "lr_shift(d0)_DISCRIM": {"mean": m_disc, "sd": s_disc, "vals": disc},
    },
    "per_seed_lifts": {
        "delayed(d50+lrs - d50)": delayed_per_seed,
        "generic_H2(d0+lrs - d0)": generic_per_seed,
        "alignment_H3": align_per_seed,
    },
    "means": {"delayed_lift": delayed_lift, "generic_H2": generic_h2,
              "alignment_H3": align_h3, "H2_frac": h2_frac, "H3_frac": h3_frac},
    "alignment_paired_t": {"t": t_align, "p_one_sided": p_align,
                           "df": 2, "signs": [1 if x > 0 else -1 for x in align_per_seed]},
    "val_ppl": {"d0": msd(d0_p)[:2], "lr_shift(d0)": msd(disc_p)[:2]},
    "11M_precedent": ELEVEN,
    "sensitivity_as_logged_seed1_null": sens,
    "VERDICT": verdict,
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "h2h3_result.json"), "w") as f:
    json.dump(out, f, indent=2)


def f3(x): return f"{x:.3f}"


lines = [
    "# H2/H3 discriminator at 23M (confirmatory size)", "",
    "Is `lr_shift` a generic optimizer booster, or is its benefit specific to",
    "the delayed model's exposure profile? Same intervention, applied to the",
    "EARLY (d0) run, forked from the identical step-2500 checkpoints.", "",
    "## Endpoints (step 5000, arith exact-match), mean ± SD over seeds 1-3", "",
    "| condition | endpoint | per-seed |",
    "|:--|--:|:--|",
    f"| d0 (early anchor) | {f3(m_d0)} ± {f3(s_d0)} | {d0} |",
    f"| d50 (delayed null) | {f3(m_d50)} ± {f3(s_d50)} | {d50} |",
    f"| d50 + lr_shift (reopened) | {f3(m_lrs)} ± {f3(s_lrs)} | {lrs} |",
    f"| **d0 + lr_shift (DISCRIMINATOR)** | **{f3(m_disc)} ± {f3(s_disc)}** | {disc} |",
    "", "## Decomposition (each lift over its OWN null)", "",
    "| component | mean | per-seed |",
    "|:--|--:|:--|",
    f"| delayed lift = lr_shift(d50) − d50 | {delayed_lift:+.3f} | {[round(x,3) for x in delayed_per_seed]} |",
    f"| generic H2 = lr_shift(d0) − d0 | {generic_h2:+.3f} | {[round(x,3) for x in generic_per_seed]} |",
    f"| alignment H3 = delayed − generic | {align_h3:+.3f} | {[round(x,3) for x in align_per_seed]} |",
    "",
    f"H3 share: {100*h3_frac:.0f}%   H2 share: {100*h2_frac:.0f}%",
    f"Alignment paired t(2) = {t_align:.3f}, one-sided p = {p_align:.4f}, "
    f"signs {[1 if x > 0 else -1 for x in align_per_seed]}",
    "",
    f"11M precedent (exploratory): delayed {ELEVEN['delayed_lift']:+.3f} = "
    f"generic {ELEVEN['generic_H2']:+.3f} + alignment {ELEVEN['alignment_H3']:+.3f} "
    f"({100*ELEVEN['H3_frac']:.0f}% H3)",
    "",
    f"Sensitivity — pairing seed 1 against the as-logged null "
    f"({d50_forkline[0]:.3f}, the model lr_shift was forked from) instead of the "
    f"canonical {D50_CANONICAL_SEED1:.3f}: delayed {sens['delayed_lift']:+.3f}, "
    f"alignment {sens['alignment_H3']:+.3f} ({100*sens['H3_frac']:.0f}% H3). "
    f"The verdict does not depend on the choice.",
    "", "## Verdict", "", verdict, "",
    "## Held-out perplexity (does the discriminator arm pay a cost?)", "",
    f"d0 {f3(msd(d0_p)[0])} ± {f3(msd(d0_p)[1])} → d0+lr_shift "
    f"{f3(msd(disc_p)[0])} ± {f3(msd(disc_p)[1])}",
]
with open(os.path.join(OUT_DIR, "h2h3_result.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
