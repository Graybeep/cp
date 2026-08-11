# Syntax LD status re-analysis — Steps 1 & 2 pre-flight

Date: 2026-07-24. Purpose: before spending any new compute, test whether the syntax
long-distance (LD) wh-licensing null is a *real* absence of a critical period, or an
artifact of looking only at the floored endpoint. Two on-disk analyses:

- **Step 1** — learning-speed / sample-efficiency re-analysis (free, no new runs).
- **Step 2 pre-flight** — feasibility of the rate-based P(high) delay test *before* launching it.

Data: `logs/syntax_delay{0,25,50,75}_exp10_size23M_seed{1,2,3}/metrics.csv`
(live LD licensing-interaction, 100-step resolution) + the 24-seed delay-0 23M
continuation-scored distribution already in `logs/szsweep_eval/gate_result.json`.

---

## STEP 1 — learning dynamics: NULL (no hidden timing effect)

Total syntax exposure is matched at **8.192M skill tokens** across all delays (design
confirmed via `realized_exposure.json`); only timing/density differ. So a critical period
could only hide in the *approach*, not the endpoint. It doesn't:

**Post-onset LD slope (per 1e6 skill tokens), seeds 1–3:**

| delay | slope mean ± SD | end LD mean ± SD |
|------:|:---------------:|:----------------:|
| 0  | +0.0017 ± 0.0031 | 0.519 ± 0.018 |
| 25 | +0.0008 ± 0.0005 | 0.517 ± 0.007 |
| 50 | +0.0001 ± 0.0010 | 0.517 ± 0.004 |
| 75 | −0.0005 ± 0.0012 | 0.523 ± 0.006 |

- Slopes are ~0 and statistically indistinguishable across delays: Welch delay0-vs-delay50
  **p = 0.46**, one-way ANOVA across all four **p = 0.48**.
- The single largest LD slope (delay0-seed1, +0.0052) is the already-known high-acquisition
  outlier seed; drop it and delay0 is flat too. **There is no LD learning signal in any
  condition for a timing effect to act on** — LD hugs the ~0.51–0.54 floor throughout.
- **Sample-efficiency view** (LD at matched skill-token budgets 1M→8M): delay0 vs delay50
  indistinguishable at *every* budget (all Welch p > 0.08; most > 0.5). No separation, no
  ordering by delay. If anything the nominal ordering at low budgets is *reversed* (density).
- Contrast: the **local** licensing interaction *does* learn (slope +0.005–0.014, climbs
  0.53→0.60) — confirming the metric can move when there's a skill to acquire. LD simply
  isn't being acquired, delay or no delay.

**Conclusion:** the floored LD endpoint does **not** hide a timing effect in the dynamics.

Artifacts: `per_run_summary.csv`, `ld_at_matched_budget.csv`, `ld_dynamics.png`,
`scripts/syntax_ld_dynamics.py`.

---

## STEP 2 pre-flight — rate-based P(high) delay test is UNDERPOWERED at n=8

The size-sweep already scored 24 delay-0 seeds at 23M with the continuation metric — that is
step 2's baseline rate, free on disk. At the pre-registered threshold band:

| threshold | P(high \| delay0, 23M) |
|----------:|:----------------------:|
| 0.55 | 5/24 = 0.208 |
| **0.56** (derived μ_low+3σ_low = 0.559) | **2/24 = 0.083** |
| 0.57 | 1/24 = 0.042 |

The high-acquisition mode is *rare at baseline* (this is exactly why the size-sweep
DISCONFIRMED reposition). A rate that low cannot fall far enough under delay for **n=8** to
register — even in the best case where delay **completely abolishes** the mode (0/8):

- Best-case one-sided Fisher p at n=8: **0.21** (thr 0.55) / **0.56** (thr 0.56). Not callable.
- Minimum delay-50 n to reach p < 0.05 *assuming total abolition*: **n ≥ 18** (thr 0.55),
  **n ≥ 82** (thr 0.56), unreachable at thr 0.57.
- And if delay's true rate ≈ delay0's rate (no effect), **no n** separates them.

So n=8 as specified is ~2–10× underpowered. Making the rate test callable needs n≈18
(permissive threshold, best case only) to n≈82 (primary threshold) delay-50 runs — i.e. the
kind of "substantial new compute for a now-secondary skill" the plan says to gate.

---

## Bottom line

Three independent looks at syntax LD at 23M now all come back empty:
1. **Mean** endpoint — null (Detect, matched exposure).
2. **Dynamics** — null (step 1, this analysis).
3. **Rate of high draws** — baseline mode is only ~8% (size-sweep); a delay could only
   lower it, and detecting that needs n≈18–82, not 8.

"Long-distance wh-extraction isn't robustly learnable in this data/scale regime — confirmed
via mean, dynamics, and (baseline) rate" is a complete, defensible conclusion. The remaining
call is whether to spend the n≈18–82 delay-50 runs to convert the rate look from "baseline is
too rare" into a formally-powered null, or to settle LD here. User decision.
