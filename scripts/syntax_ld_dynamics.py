"""
Step 1 (LD status re-analysis): learning-speed / sample-efficiency re-analysis of
long-distance (LD) wh-licensing across delay conditions, using ONLY data already on disk.

Motivation: the Detect verdict for syntax LD is a NULL on the *endpoint* (all delays
floor at ~0.51-0.54 licensing-interaction; matched total skill exposure = 8.192M tokens).
Question: does a floored endpoint hide a real *timing* effect in the learning DYNAMICS
(slope / sample-efficiency), rather than final accuracy?

Data source: logs/syntax_delay{0,25,50,75}_exp10_size23M_seed{1,2,3}/metrics.csv
  col 'syntax_accuracy'        = LD licensing interaction  (mean of with_gap_LD, no_gap_LD)
  col 'syntax_accuracy_local'  = local licensing interaction
  col 'skill_tokens_seen'      = cumulative syntax-skill tokens (matched to 8.192M at end)
Resolution: every 100 steps, 50 points/run. This is the ONLY high-res trajectory that
exists for all runs (intermediate checkpoints were not saved, so the primary
continuation-scored metric cannot be re-scored across the trajectory).

Two x-axes:
  (1) training step        -> raw dynamics
  (2) skill_tokens_seen    -> sample-efficiency (fair across delays: all traverse 0->8.192M)
The sample-efficiency view is the critical-period test: at a MATCHED skill-token budget,
does earlier exposure reach higher LD? Endpoint is matched by construction, so only the
approach can differ.

Caveat (density confound): delay conditions also differ in skill DENSITY (0.1/0.13/0.2/0.4),
so "efficiency per skill token" conflates timing with density. This is a first look, not a
clean decomposition (see density-decoupling-control note). Reported honestly below.
"""
import os, glob, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "syntax_ld_dynamics")
os.makedirs(OUT, exist_ok=True)

DELAYS = [0, 25, 50, 75]
SEEDS = [1, 2, 3]
TOTAL_SKILL = 8_192_000
# Common skill-token budgets to interpolate LD at (fair cross-delay comparison points)
BUDGETS = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6]

def load(delay, seed):
    f = os.path.join(ROOT, "logs", f"syntax_delay{delay}_exp10_size23M_seed{seed}", "metrics.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    return df

def interp_at_budget(df, budget, ycol):
    """LD value when skill_tokens_seen first reaches `budget` (linear interp)."""
    x = df["skill_tokens_seen"].values.astype(float)
    y = df[ycol].values.astype(float)
    if x.max() < budget:
        return np.nan
    # np.interp requires increasing x; skill_tokens_seen is monotone nondecreasing
    return float(np.interp(budget, x, y))

def active_slope(df, ycol):
    """OLS slope of LD vs skill_tokens_seen over the post-onset region (skill_tokens>0),
    in units of LD per 1e6 skill tokens."""
    m = df["skill_tokens_seen"].values.astype(float) > 0
    x = df["skill_tokens_seen"].values.astype(float)[m] / 1e6
    y = df[ycol].values.astype(float)[m]
    if len(x) < 3:
        return np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)

rows = []
budget_rows = []
for d in DELAYS:
    for s in SEEDS:
        df = load(d, s)
        if df is None:
            print(f"MISSING delay{d} seed{s}")
            continue
        end_ld = float(df["syntax_accuracy"].iloc[-1])
        end_local = float(df["syntax_accuracy_local"].iloc[-1])
        peak_ld = float(df["syntax_accuracy"].max())
        peak_local = float(df["syntax_accuracy_local"].max())
        # baseline LD at onset (first row where skill_tokens>0) -> "start" of learning
        onset_idx = (df["skill_tokens_seen"].values > 0).argmax()
        start_ld = float(df["syntax_accuracy"].iloc[onset_idx])
        rows.append(dict(
            delay=d, seed=s,
            end_ld=end_ld, peak_ld=peak_ld, start_ld=start_ld,
            ld_gain=end_ld - start_ld,
            slope_ld_per_1e6=active_slope(df, "syntax_accuracy"),
            slope_local_per_1e6=active_slope(df, "syntax_accuracy_local"),
            end_local=end_local, peak_local=peak_local,
        ))
        for b in BUDGETS:
            budget_rows.append(dict(delay=d, seed=s, budget=b,
                                    ld=interp_at_budget(df, b, "syntax_accuracy"),
                                    local=interp_at_budget(df, b, "syntax_accuracy_local")))

summ = pd.DataFrame(rows)
bud = pd.DataFrame(budget_rows)
summ.to_csv(os.path.join(OUT, "per_run_summary.csv"), index=False)
bud.to_csv(os.path.join(OUT, "ld_at_matched_budget.csv"), index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

print("\n================ PER-RUN SUMMARY (LD = long-distance licensing interaction) ================")
print(summ.round(4).to_string(index=False))

print("\n================ AGGREGATE BY DELAY (mean +/- SD across seeds 1-3) ================")
agg = summ.groupby("delay").agg(
    end_ld_mean=("end_ld","mean"), end_ld_sd=("end_ld","std"),
    peak_ld_mean=("peak_ld","mean"), peak_ld_sd=("peak_ld","std"),
    slope_ld_mean=("slope_ld_per_1e6","mean"), slope_ld_sd=("slope_ld_per_1e6","std"),
    slope_local_mean=("slope_local_per_1e6","mean"), slope_local_sd=("slope_local_per_1e6","std"),
).reset_index()
print(agg.round(4).to_string(index=False))

print("\n============ LD AT MATCHED SKILL-TOKEN BUDGET (sample-efficiency view) ============")
print("mean LD across seeds 1-3 at each budget; if early exposure is more sample-efficient,")
print("delay0 column should sit ABOVE delay50/75 at intermediate budgets and converge at 8e6.\n")
piv = bud.groupby(["budget","delay"])["ld"].mean().unstack("delay")
piv.index = [f"{int(b/1e6)}M" for b in piv.index]
print(piv.round(4).to_string())

print("\n  (local licensing interaction, same view)\n")
pivl = bud.groupby(["budget","delay"])["local"].mean().unstack("delay")
pivl.index = [f"{int(b/1e6)}M" for b in pivl.index]
print(pivl.round(4).to_string())

# ---- Statistical read: is the post-onset LD slope different across delays? ----
print("\n================ SLOPE TEST (LD per 1e6 skill tokens, post-onset) ================")
from scipy import stats
groups = [summ[summ.delay==d]["slope_ld_per_1e6"].values for d in DELAYS]
for d, g in zip(DELAYS, groups):
    print(f"  delay{d}: slopes = {np.round(g,4)}  mean={np.mean(g):.4f} sd={np.std(g,ddof=1):.4f}")
# delay0 vs delay50 Welch (primary contrast used elsewhere in project)
g0 = summ[summ.delay==0]["slope_ld_per_1e6"].values
g50 = summ[summ.delay==50]["slope_ld_per_1e6"].values
t, p = stats.ttest_ind(g0, g50, equal_var=False)
print(f"\n  Welch delay0 vs delay50 slope: t={t:.3f} p={p:.4f}")
# one-way across all four delays
f, pf = stats.f_oneway(*groups)
print(f"  One-way ANOVA across delays 0/25/50/75: F={f:.3f} p={pf:.4f}")

# ---- LD at matched budget: delay0 vs delay50 at each budget ----
print("\n============ MATCHED-BUDGET CONTRAST delay0 vs delay50 (Welch per budget) ============")
for b in BUDGETS:
    a = bud[(bud.delay==0)&(bud.budget==b)]["ld"].dropna().values
    c = bud[(bud.delay==50)&(bud.budget==b)]["ld"].dropna().values
    if len(a)>=2 and len(c)>=2:
        t,p = stats.ttest_ind(a,c,equal_var=False)
        print(f"  budget {int(b/1e6)}M: delay0={a.mean():.4f}+/-{a.std(ddof=1):.4f}  "
              f"delay50={c.mean():.4f}+/-{c.std(ddof=1):.4f}  Welch p={p:.4f}")

print(f"\nOutputs written to {OUT}")

# ---- Plot ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = {0:"#2166ac", 25:"#66c2a5", 50:"#f4a582", 75:"#b2182b"}

    for (metric, title, axrow) in [("syntax_accuracy","LD (long-distance)",0),
                                    ("syntax_accuracy_local","local",1)]:
        ax_step = axes[axrow,0]; ax_tok = axes[axrow,1]
        for d in DELAYS:
            curves_step = []; curves_tok = []
            for s in SEEDS:
                df = load(d,s)
                if df is None: continue
                ax_step.plot(df["step"], df[metric], color=colors[d], alpha=0.18, lw=1)
                ax_tok.plot(df["skill_tokens_seen"]/1e6, df[metric], color=colors[d], alpha=0.18, lw=1)
                curves_step.append(df.set_index("step")[metric])
            # seed-mean vs step
            mean_step = pd.concat(curves_step, axis=1).mean(axis=1)
            ax_step.plot(mean_step.index, mean_step.values, color=colors[d], lw=3, label=f"delay{d}")
            # seed-mean vs budget (interpolated)
            ys = []
            for b in BUDGETS:
                v = bud[(bud.delay==d)&(bud.budget==b)]["ld" if metric=="syntax_accuracy" else "local"].mean()
                ys.append(v)
            ax_tok.plot([b/1e6 for b in BUDGETS], ys, color=colors[d], lw=3, marker="o", label=f"delay{d}")
        ax_step.axhline(0.5, color="k", ls=":", lw=1, alpha=0.6)
        ax_tok.axhline(0.5, color="k", ls=":", lw=1, alpha=0.6)
        ax_step.set_title(f"{title}: acc vs training step"); ax_step.set_xlabel("step"); ax_step.set_ylabel(title)
        ax_tok.set_title(f"{title}: acc vs skill tokens seen (sample-efficiency)")
        ax_tok.set_xlabel("skill tokens seen (millions)"); ax_tok.set_ylabel(title)
        ax_step.legend(fontsize=11); ax_tok.legend(fontsize=11)
    fig.suptitle("Syntax LD/local learning dynamics across delay conditions (23M, seeds 1-3)", fontsize=18)
    fig.tight_layout(rect=[0,0,1,0.97])
    p = os.path.join(OUT, "ld_dynamics.png")
    fig.savefig(p, dpi=130)
    print(f"Plot: {p}")
except Exception as e:
    print(f"[plot skipped: {e}]")
