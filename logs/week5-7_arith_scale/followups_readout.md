# Week 5-7 arith scale-check — FOLLOW-UPS read-out (n=3)

**COMPLETE 2026-07-24 01:00.** All follow-up runs finished. Verdicts are mechanical
(`scripts/arith_scale_gate.py` → `gate_result.json`); rules pre-registered in
`notes/week5-7-arith-scale-check-preregistration.md` + the two hardenings in
`memory/week5-7-arith-scale-check.md`. Metric = `arith_accuracy` @ step 5000.

The probe (seed-1, n=1) STOPPed 11M at Rule-1 FLOOR and flagged a forgetting-gate
breach at 60M. These follow-ups (a) fill 60M to n=3, (b) add a no-op drift control to
decompose that breach, and (c) run the exploratory 11M H1/H2/H3 discriminator. The
11M pre-registered cell stays STOPped — the 11M runs here are a separately-labelled
exploratory follow-up, not a reopened cell.

---

## (A) 60M DETECT — Rule 1b, n=3

| arm | seed1 | seed2 | seed3 | mean ± SD |
|---|---|---|---|---|
| d0 (early anchor) | 0.798 | 0.840 | 0.848 | **0.829 ± 0.027** |
| d50 (delayed null) | 0.574 | 0.646 | 0.580 | **0.600 ± 0.040** |

**gap = 0.229**, one-sided Welch t=8.23, p=0.0010 → **CP CONFIRMED at 60M.**

Scale-stable Detect finding: 23M gap 0.214 ± 0.055 vs 60M gap 0.229 — essentially
identical across a 2.6× size difference. The arithmetic critical period is not a
23M-local artifact.

## (B) 60M REOPEN — Rule 2, n=3 + forgetting gate

| arm | seed1 | seed2 | seed3 | mean ± SD |
|---|---|---|---|---|
| lr_shift (fork d50@2500) | 0.796 | 0.818 | 0.774 | **0.796 ± 0.022** |

- **2a** gap 0.229 ≥ 0.15 → PASS (capacity guard cleared)
- **2b** 0.796 ≥ 0.90 × d0 (0.746) → PASS
- **2c** lift 0.196 ≥ 0.10 → PASS
- → **THRESHOLD reopening GENERALIZES** on the skill axis.
- **Same-endpoint band:** lr_shift mean 0.796 ∈ [0.722, 0.812], SD 0.022 ≤ 0.045 →
  IN-BAND, precise. Overlaps 23M's 0.767 → **recovered endpoint is scale-invariant**
  (extends Week-4's "pins ~0.77 regardless of null depth" across a 2.6× size step).
  Not > 0.812, so NOT sub-case (c) "endpoint scales with capacity."

**BUT the forgetting gate BREACHES at n=3, all three seeds:**

| | seed1 | seed2 | seed3 | mean |
|---|---|---|---|---|
| c − b (lr_shift ppl − d50 null ppl) | +1.790 | +1.601 | +1.619 | **+1.670 > +1.0** |

Per the CLAUDE.md operational definition, recovery bought with this general-LM cost
fires the gate as **forgetting, not reopening** — and the gate is honored as written
(no post-hoc retuning). BUT see (C): a matched-LR comparison at **n=3** shows the
intervention arm is ahead of the null by **2.48 ± 0.24 ppl** when both are read at the
same LR — the arm is measured mid-anneal at 6× the null's terminal LR, and that lead
*exceeds* the 1.67 breach. So the +1.67 is **BOUNDED-ARTIFACT**: probably a horizon
artifact, with any genuine residual cost bounded above by the breach and needing the
arm-extension to confirm. Honest 60M verdict: full, scale-invariant *skill* recovery
(0.796, ≈99.7% of the 0.798 anchor) with an **apparent** general-LM cost that (i)
mechanically trips the pre-registered gate at the fixed 5000-step horizon [gate honored
as written], but (ii) is confounded by LR-anneal truncation across all 3 seeds and
leans toward (near-)full erasure on extension (residual bounded ≤ 1.67; no specific
endpoint quoted — slope unmeasured). Reported as a split with that n=3 caveat
foregrounded, not as clean forgetting.

## (C) 60M DRIFT CONTROL — is the +1.67 ppl real or fork mechanics?

No-op fork (`ivlrshift2501at2501`: source_start == at_step == 2501 → unmodified LR),
same fork machinery / checkpoint / data stream as lr_shift. If fork mechanics alone
moved ppl, this arm would drift from the null too.

| Δ vs d50 null | seed1 | seed2 | seed3 | mean |
|---|---|---|---|---|
| val_ppl | −0.032 | −0.008 | +0.005 | **−0.012** |
| arith_acc | +0.002 | −0.006 | +0.038 | **+0.011** |

Fork mechanics contribute **~0** on both ppl and accuracy — the no-op fork reproduces
the null. Therefore the **+1.67 ppl is intervention-induced** (the replayed high-LR
warmup), NOT fork machinery. This cleanly kills interpretation (a). Of the two
remaining candidates in the batch header:
- **(b)** miscalibrated absolute gate where 60M's delay cost b−a ≈ 0 — live; flagged,
  gate applied as written (not retuned post-hoc), breach survives the relative reading
  too (mean 1.670 / mean null ppl 41.73 = ~4.0% vs the gate's ~1.97%).
- **(c)** HORIZON ARTIFACT — **BOUNDED-ARTIFACT at n=3** (mechanical:
  `arith_scale_gate.py` → `matched_lr_test`). `ShiftedLRWindow(source_start=0)` replays
  the early schedule over [2501,5000], so at step 5000 the arm sits at LR 3.57e-4 (6×
  the null's annealed 6.0e-5) — cut off *mid-anneal*. Compared to the null **at matched
  LR** (its step 2500, schedule is deterministic so the phase is identical per seed):

  | seed | arm@5000 (LR/ppl) | null@matchedLR (step/ppl) | null@5000 | matched-LR lead | breach c−b |
  |---|---|---|---|---|---|
  | 1 | 3.57e-4 / 43.03 | 2500 / 45.23 | 41.24 | **+2.20** | +1.79 |
  | 2 | 3.57e-4 / 43.51 | 2500 / 46.14 | 41.91 | **+2.63** | +1.60 |
  | 3 | 3.57e-4 / 43.68 | 2500 / 46.28 | 42.06 | **+2.60** | +1.62 |
  | **mean** | | | | **+2.48 ± 0.24** | **+1.67 ± 0.11** |

  Direction AND magnitude replicate across all 3 seeds — this is not a single-seed
  mechanistic aside. The arm is ahead of the null by **2.48 ± 0.24 ppl at matched LR**,
  and that lead **exceeds the breach (1.67)** it would need to explain. Quantitative
  bound: the null gained ~4.15 ppl finishing its anneal (matched-LR → floor); a genuine
  general-LM cost persists **only if** the arm's (unmeasured) remaining anneal delivers
  **< 1.67 ppl** — i.e. < ~40% as effective as the null's ~4.15, despite the arm being
  *ahead* at matched LR. So the evidence leans toward the residual being closer to zero
  than to the full breach.

  **No specific extrapolated endpoint is quoted.** What is measured (n=3) is the arm's
  *starting position* for the remaining anneal (2.48 ± 0.24 ahead at matched LR); the
  anneal *slope* is not measured, so a same-slope projection to a specific arm endpoint
  is NOT earned and is deliberately kept out of the prose — it would imply the
  intervention slightly *outperforms* the standard schedule on general capability, an
  interesting-if-true secondary claim the current data cannot support. The qualitative
  bound (residual ≤ 1.67, evidence pointing toward the low end) carries the finding
  without it.

  **Named status: BOUNDED-ARTIFACT, not a binary artifact/forgetting call.** Measured
  (n=3): the matched-LR lead caps any real residual cost below the breach and leans
  toward near-full erasure. Not measured: whether the arm anneals as efficiently as the
  null — so "part-artifact, part-real residual (≤ the breach)" is bounded above but not
  excluded. The arm-extension is what would collapse this to a point estimate (and is
  the only thing that could earn a specific endpoint).

  **A decisive confirmation still needs the arm extended past 5000 so its replayed
  cosine finishes annealing — genuinely non-trivial to wire, NOT the cheap path:**
  `total_steps` is in the cosine denominator (`train.py:52`) and resuming with a
  changed `total_steps` is *asserted against* (`train.py:166`, "would change the LR
  schedule"), so a naive `--total-steps` override is INVALID (rescales LR over the
  measured window). A correct extension needs a schedule mode that lets the replayed
  window finish annealing past `total_steps` while the base schedule stays fixed, plus
  relaxing the `intervention is None` fork guard (`run_intervention.py:63`) to re-fork
  from the arm's `step_5000`. The cheap null-extension the design might suggest is
  **uninformative**: the null is already at floor LR (6.0e-5) and flat by 5000, so
  running it further shows nothing and doesn't speak to the arm's high-LR situation.

## (D) 11M H1/H2/H3 DISCRIMINATOR — exploratory (pre-registered cell stays STOPped)

| arm | mean acc (n=3) | per-seed |
|---|---|---|
| d0 (early anchor) | 0.538 | 0.556 / 0.538 / 0.520 |
| d50 (null) | 0.357 | 0.304 / 0.332 / 0.436 |
| d50 + lr_shift | 0.747 | 0.726 / 0.768 / 0.746 |
| **d0 + lr_shift (discriminator)** | **0.653** | 0.612 / 0.704 / 0.644 |

The correct H2/H3 test compares lr_shift's lift on each run **over that run's OWN
null**, not the gap between the two boosted endpoints:
- delayed lift (the reopening effect): d50+lrs − d50 = 0.747 − 0.357 = **+0.389**
- generic component (H2): d0+lrs − d0 = 0.653 − 0.538 = **+0.115** (per-seed 0.056 /
  0.166 / 0.124 — real but seed-variable)
- alignment component (H3): 0.389 − 0.115 = **+0.274**  →  0.115 + 0.274 = 0.389 ✓

→ **H3 DOMINANT (exposure-alignment): ~70% of the reopening effect requires the
delayed-exposure profile, ~30% is a generic optimizer boost.** lr_shift lifts the
delayed run (+0.389) ~3.4× more than the early run (+0.115), so the gain is
exposure-timing-specific — high LR replayed over the arith window — **not** mere
accuracy inflation.

*(Correction: an earlier pass mislabeled this H2-dominant by using d50+lrs − d0+lrs
(0.093) as the "delayed residual." That quantity compares two boosted endpoints with
different baselines and is not the H2/H3 discriminator; 0.115 + 0.093 = 0.208 also
failed to reconcile with the observed ~0.39 delayed lift. The correct decomposition
sums to 0.389 and flips the verdict to H3.)*

This does **not** overturn the existence-proof point (an 11M model *can* represent the
skill at ~0.73) and does **not** reopen the Rule-1 FLOOR STOP on the pre-registered 11M
cell. It clarifies the mechanism: at 11M the lr_shift gain is genuinely exposure-aligned
reopening (as at 23M/60M), not a generic number-inflation — consistent with Week-4's
"d50 arith closure is largely LR-schedule closure." The d0 anchor's own shortfall
(never plateaued, LR-starved under standard cosine at 11M) remains a separate
capacity/budget-axis matter, untouched by this.

---

## BOTTOM LINE

1. **Detect is scale-robust.** Arithmetic CP confirmed at 60M (gap 0.229, p=0.0010),
   matching 23M (0.214) — and, from the probe, *absent-as-measured* only at 11M where
   the anchor itself is LR-starved (FLOOR, a capacity/budget axis, not a timing claim).
2. **Reopen skill-recovery is scale-invariant** — 60M lr_shift lands 0.796, inside the
   same-endpoint band around 23M's 0.767, clearing all three Rule-2 conditions.
3. **The 60M forgetting-gate breach (c−b = +1.67, all seeds) is real at the fixed
   horizon but BOUNDED-ARTIFACT at n=3.** The drift control rules out fork mechanics;
   the matched-LR test (all 3 seeds) shows the arm ahead of the null by **2.48 ± 0.24
   ppl** at the same LR — a lead that exceeds the breach — because the arm is measured
   mid-anneal at 6× the null's terminal LR. Any genuine residual cost is bounded above
   by the breach, with the evidence leaning toward the low (near-zero) end; no specific
   extrapolated endpoint is claimed (anneal slope unmeasured). Verdict: **full
   scale-invariant skill recovery with an apparent, probably-transient (bounded)
   general-LM cost.** Decisive confirmation needs the arm extended past 5000 — a
   non-trivial build (naive `--total-steps` is invalid; see (C)); the cheap
   null-extension is uninformative (null already annealed).
4. **11M lr_shift is exposure-alignment reopening (H3, ~70%), not a generic booster.**
   It lifts the delayed run ~3.4× more than the early run; the FLOOR STOP still stands.
