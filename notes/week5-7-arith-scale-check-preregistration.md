# Week 5-7 arithmetic scale-check — pre-registration

**Written 2026-07-22, before any run.** Batch:
`scripts/week5-7_arith_scale_check_batch.sh`. This resolves the scale-asymmetry:
syntax got a full 3-size sweep and failed it (dropped from the grid), while
arithmetic's Detect gap and lr_shift reopening have only ever been measured at
23M. We test both at **11M and 60M** so the kept finding clears the same
scale-robustness bar we used to reject syntax.

Everything a reader would otherwise "eyeball" is fixed here as a NUMBER, keyed to
the established 23M result. No threshold below is invented for these runs — each
is derived from and passes at 23M.

## The 23M reference (n=3, reconciled — `logs/week4_close/results.md`)
| quantity | value |
|---|---|
| d0 early anchor | **0.791 ± 0.007** (0.798 / 0.792 / 0.784) — seed-STABLE |
| d50 delayed null | **0.577 ± 0.051** (0.524 / 0.626 / 0.582) — seed-VARIABLE |
| Detect gap (d0 − d50) | **0.214 ± 0.055**; per-seed 0.274 / 0.166 / 0.202; **min-seed 0.166** |
| lr_shift treated | **0.767 ± 0.014** (0.752 / 0.778 / 0.772) — tight |
| recovery lift (lr_shift − d50) | **~0.190** (0.228 / 0.152 / 0.190) |
| 90%-of-anchor gate | 0.90 × 0.791 = **0.712** (orig pilot used 0.90 × 0.798 = 0.718) |

Metric = arith exact-match accuracy at step 5000, read from
`logs/<run_id>/metrics.csv` (column `arith_accuracy`), exactly as the seeds-2-3
replication did. Forgetting check = general `val_ppl` (intervention cost ≤ +1.0).

## Design (per size ∈ {11M, 60M}; seeds match 23M: 1,2,3)
Three arms, sharing compute (fork reuses the d50 step-2500 checkpoint):
- **d0** — arith delay-0 (early anchor).
- **d50** — arith delay-0.5 (delayed null AND lr_shift fork source @2500).
- **lr_shift** — fork d50@2500, replay early LR segment over [2500,5000).

Cost (measured from the syntax sweep: 11M ≈13 min/run, 60M ≈45 min/run to 5000;
each size/seed = 2 full + 1 half run = 2.5 run-eq):
| stage | seeds | 11M | 60M | total |
|---|---|---|---|---|
| probe | 1 | 32.5 min | 112.5 min | **~2.4 h** |
| fill  | 2,3 | 97 min | 338 min | **~4.8 h** |

---

## RULE 1 — n=1 probe: does the pattern hold *directionally* (go/no-go to fill n=3)?
Applied **per size** to the seed-1 probe. Two numbers, not an eyeball read:
- **d0 anchor floor:  d0 ≥ 0.60.**  Below this the model has not robustly
  acquired arithmetic even with early exposure → no headroom for a delay gap
  (the thin-headroom trap that killed long-distance syntax).
- **probe gap:  (d0 − d50) ≥ 0.10.**  Deliberately *below* the n=3 claim bar
  (Rule 2's 0.15): the probe is one seed and 23M's own gap ranges down to 0.166
  across seeds, so a single new-size seed could legitimately draw low — the
  probe must not false-negative a real effect on one unlucky seed. 0.10 is still
  ≫ the ~±0.02 within-seed run noise.

Decision tree (per size):
| probe read | verdict | action |
|---|---|---|
| d0 < 0.60 | **FLOOR / capacity-limited** | STOP at this size. Report "arith acquisition capacity-limited at {size}; CP undetectable (no headroom)" — a capacity boundary, NOT a CP-absence claim. Do not fill. |
| d0 ≥ 0.60 AND gap ≥ 0.10 | **CP present (directional)** | GO — `... fill` to n=3 (Detect + Reopen). |
| d0 ≥ 0.60 AND 0.05 ≤ gap < 0.10 | **SUB-THRESHOLD GAP — indeterminate at n=1** (named verdict, see Rule 1b) | GO anyway — fill to n=3; n=1 can't separate a shallow-seed draw from a true small gap. n=3 adjudicates via Rule 1b. |
| d0 ≥ 0.60 AND gap < 0.05 | **CP absent (directional)** | Confirm with +2 seeds on d0/d50 ONLY (skip lr_shift — nothing to reopen). If n=3 holds gap < 0.05 → finding: "CP absent at {size}" = scale-dependent CP. |

### Reporting a d0-floor failure (the 11M risk): a capacity boundary, NOT "CP absent"
A floor failure (d0 < 0.60) is **a reportable positive finding on a different
axis**, not a null to discard — and must never be flattened into "arithmetic's
critical period doesn't hold at small scale." Two orthogonal questions:
- **Can the skill be learned here at all?** (capacity axis — measured by d0)
- **Does timing of exposure matter?** (critical-period axis — measured by the gap)

A floor failure answers only the first, in the negative, which makes the second
**undetectable** — there is no headroom to even pose the timing question.

Pre-registered phrasing (use verbatim):
> "At 11M the model does not acquire arithmetic even under early exposure
> (d0 = X < 0.60), so the critical-period question is **undetectable at this
> capacity — not answered in the negative**. This is a capacity/floor boundary,
> orthogonal to exposure timing: the model cannot learn the skill at all at this
> size, independent of when it is introduced. It is therefore evidence about a
> minimum-capacity threshold for the skill, **not** evidence that arithmetic's
> critical period weakens or vanishes at small scale."

This is the exact capacity-undetectable-vs-CP-absent distinction that parallels
syntax's DISCONFIRM-vs-clean-null care. "CP absent" is a *timing* claim reserved
for the Rule-1 gap<0.05 branch (d0 ≥ 0.60, skill learned, timing shown not to
matter). A floor failure is never "CP absent."

> **POST-HOC AMENDMENT 2026-07-23, after the 11M probe — scope of the floor
> phrasing only. NO VERDICT CHANGES.** The verbatim phrasing above makes
> unqualified *architecture-level* claims — "the model cannot learn the skill at
> all at this size" and "evidence about a minimum-capacity threshold for the
> skill." The 11M probe falsifies those as written: the 11M lr_shift arm scores
> **0.726**, so an 11M model representing arithmetic at 0.726 demonstrably
> exists, and no claim of the form "11M cannot exceed 0.60" is tenable.
>
> The amendment is strictly one of **scope**, and it changes nothing operational:
> - **The Rule-1 STOP at 11M stands.** Rule 1 gates on the **d0 anchor under the
>   standard schedule**, because the Detect gap is d0 − d50 and both must be
>   standard-schedule runs for the subtraction to mean anything. lr_shift is not
>   a standard-schedule run and is not evidence about d0's own ceiling. The
>   design rationale for the floor is untouched.
> - **Replacement phrasing (use this instead of the verbatim block above):**
>   *"At 11M the early-exposure baseline does not reach criterion within the
>   matched 5000-step budget under the standard cosine schedule (d0 = X < 0.60),
>   so the critical-period question is undetectable in this configuration — not
>   answered in the negative. This is a floor on the achievable anchor at matched
>   compute and schedule; it is NOT a claim about the architecture's
>   representational capacity at this size."*
>
> That this amendment leaves every verdict unchanged is the point: it narrows an
> over-claiming interpretive sentence, and is not a rescue of an unwanted result.

## RULE 1b — the n=3 Detect verdict band (0.05 ≤ gap < 0.15): **ATTENUATED CP**
*Added 2026-07-23, before any run.* Rule 1's bands are **fill decisions at n=1**;
Rule 2's 2a (gap ≥ 0.15) is a **Reopen** admission bar. Between them the *Detect*
verdict at n=3 had no name for a gap that clears the CP-absent cutoff (0.05) but
misses 2a — a size could land at gap = 0.12 and be described only as "2a fails,
reopening not assessable," which silently reads as "no CP." That conflation is
now closed: **"2a fails" has two distinct causes and they get different names.**

Applied per size at n=3, given d0 ≥ 0.60 (capacity clears). Significance =
one-sided Welch *t* on the three d0 vs three d50 step-5000 endpoints, α = 0.05:

| n=3 gap (d0 − d50) | Welch p | **Detect verdict (named)** | Reopen status |
|---|---|---|---|
| ≥ 0.15 | (report) | **CP confirmed** — at least as clear as 23M's weakest replicate | assessable → Rule 2 |
| 0.05 ≤ gap < 0.15 | p < 0.05 | **ATTENUATED CRITICAL PERIOD** — a real CP, shallower than 23M's. A *positive* Detect finding on its own (CP severity is scale-graded), not a null. | **not assessable** (2a fails by design) |
| 0.05 ≤ gap < 0.15 | p ≥ 0.05 | **INCONCLUSIVE AT n=3** — neither CP-present nor CP-absent; the design cannot resolve a gap this small at this seed depth. | not assessable |
| < 0.05 | (report) | **CP absent at {size}** — skill learned, timing shown not to matter → scale-dependent CP | nothing to reopen |

Three consequences pre-registered now:
- **ATTENUATED ≠ CP absent.** "CP absent" stays reserved for gap < 0.05. An
  attenuated size is reported as *"a critical period is present at {size} but
  shallower (gap = X ± SD) than at 23M (0.214 ± 0.055); it falls below the
  pre-registered 0.15 admission bar, so the reopening question is not assessable
  at this size"* — never as "the CP does not hold at {size}".
- **INCONCLUSIVE is a declared non-result, and escalates — AT EITHER SIZE,
  symmetrically.** *(Resolved 2026-07-23: the escalation is a property of the
  VERDICT, not of the size.)* On an INCONCLUSIVE verdict, run +2 seeds (n=5) on
  the **d0 and d50 arms only** — lr_shift is not re-run, since 2a fails either
  way and nothing about the Reopen claim depends on it — and re-apply Rule 1b
  **once**. If still inconclusive at n=5, it stays INCONCLUSIVE in the write-up:
  an honest unresolved cell, not rounded to either neighbour.
  - Cost: **11M ≈ 0.9 h** (4 runs × ~13 min), **60M ≈ 3.0 h** (4 runs × ~45 min).
  - So an 11M read of *gap = 0.08, p = 0.06* gets **exactly the same +2 seeds**
    as the identical read at 60M. Escalating only the larger size would make the
    resolution of an ambiguous cell depend on which size produced it — an
    asymmetry with no scientific justification (and, at 11M's 0.9 h, not even a
    compute one). Both sizes escalate, or neither does.
  - The escalation is capped at one round and is triggered **only** by
    INCONCLUSIVE. It is not available to rescue a CP-absent verdict (gap < 0.05),
    an ATTENUATED verdict (already resolved), or a failed Rule-2 gate — those are
    answers, not ambiguity. The one unrelated exception already on the books is
    the same-endpoint band's *in-band-but-SD > 0.045* cell, which escalates for a
    different reason (imprecision, not indeterminacy).
- **The lr_shift arm still exists** in these branches (the `fill` stage runs all
  three arms). Report its endpoint **descriptively and labelled
  non-confirmatory**; even if it satisfies 2b and 2c, 2a's failure means no
  generalization claim of either tier may be made from it.

Power calibration (why the bands sit where they do), at 23M-observed noise
(d0 SD 0.007, d50 SD 0.05, n=3): the Welch test certifies gap = 0.10 (p = 0.036)
and 0.12 (p = 0.025) but not 0.06 (p = 0.086) or 0.08 (p = 0.053). So
ATTENUATED is reachable over roughly the upper half of the band and
INCONCLUSIVE over the lower half — which is the intended behaviour, not a flaw.
Self-consistency: 23M's own gap gives p = 0.0085 → **CP confirmed** ✓.

## RULE 2 — n=3 per-size Reopen gate WITH a capacity-limit guard
The worry: at a low-d0 size, "90% of a low anchor" is a low bar the null may
nearly meet on its own, so a *technical* pass could be mis-sold as "the
mechanism generalizes" when there was barely anything to recover. Guard against
it by requiring a real CP AND a substantive absolute move, not just the gate.

A **"reopening mechanism generalizes at {size}"** claim requires **ALL THREE**
(n=3 means):
- **(2a) Real CP to reopen:  gap (d0 − d50) ≥ 0.15.**  Calibrated just below
  23M's shallowest-seed gap (0.166) = "a CP at least as clear as 23M's weakest
  replicate." *This is the capacity-limit guard the user asked for:* a low-d0
  size yields a compressed gap that fails 2a, so it can never reach a
  generalization claim on a technicality.
- **(2b) Clears own anchor:  lr_shift ≥ 0.90 × d0_size** (per-size, never the
  23M 0.718). Same rule as 23M.
- **(2c) Substantive lift:  (lr_shift − d50) ≥ 0.10.**  The intervention closed
  a real absolute amount (23M lift ~0.190), not a technicality riding a high null.

Interpretation matrix. With **2a failing**, stop — reopening is not assessable
(NOT a generalization claim; this is where a capacity-limited technical pass is
caught) — **and name *why* it failed using Rule 1b**: ATTENUATED CP (0.05 ≤ gap
< 0.15, p < 0.05), INCONCLUSIVE (same band, p ≥ 0.05), or CP absent (gap < 0.05).
"No comparable CP" is not one bucket. With **2a passing** (a real CP exists), the two
remaining conditions define a FOUR-cell space — do NOT collapse them into one
"fails" bucket:

| 2b (endpoint ≥ 0.90×d0) | 2c (lift ≥ 0.10) | verdict |
|:---:|:---:|---|
| pass | pass | **Threshold reopening generalizes at {size}** (the strong claim). |
| **fail** | **pass** | **PARTIAL REOPENING** — a real, substantive recovery effect that falls short of the pre-registered 90% criterion. NOT "does not generalize." Directly analogous to the syntax sweep's *disconfirm-under-a-conservative-gate* (a demonstrated effect missing the strict bar), as opposed to a clean null. Report as: the reopening EFFECT transfers to {size} (lift real) but does not reach the reopening CRITERION. |
| pass | fail | **Threshold met but non-substantive** — endpoint cleared 0.90×d0 while the lift was trivial (the null already sat near the bar). The 2c guard firing on the recovery side; do NOT call this clean generalization. Narrow regime (needs high-ish d50 with 2a still ≥0.15). |
| fail | fail | **No meaningful recovery** — lr_shift does not reopen at {size}. The true negative (clean-null analogue). |

**Two-tier generalization language (use exactly).** The reopening *effect*
generalizes across a size where **2a & 2c** hold (a real CP, a real lift);
*threshold reopening* generalizes only where **2a & 2b & 2c** all hold. A
PARTIAL-REOPENING size supports the former, not the latter — report both tiers,
never one flat pass/fail.

**Evidence requirement for a PARTIAL-REOPENING verdict** (so "short because the
deficit was deep" is shown, not asserted — per the show-both-endpoints rule):
report the per-seed d0 AND d50 (deficit depth) beside the lr_shift endpoint, and
compare the lr_shift endpoint to 23M's pinned ~0.767. This separates the two
reasons 2b can be missed:
  (a) endpoint recovered to its usual ~0.77 but this size's own d0 ceiling is
      higher, so 0.90×d0 is a *raised bar* → "recovers to the usual absolute
      level, below 90% of a higher ceiling";
  (b) the endpoint genuinely landed lower than ~0.77 → "recovery itself is
      weaker at this size".
Both are PARTIAL, but they are different findings and must be named distinctly.

### OUTCOME TO FEATURE PROMINENTLY: scale-invariant recovery endpoint (60M sub-case a)
Pre-registered *before data*: if **60M** lands in PARTIAL REOPENING specifically
via **sub-case (a)** — the lr_shift endpoint recovers to ~0.767 (its usual 23M
level) but 2b fails only because 60M's higher d0 ceiling raises the 0.90×d0 bar
above it — that is **the strongest generalization result this design can
produce**, and should be *foregrounded* in the write-up, not reported as a
threshold miss.

Why it is the strongest form of "this generalizes": it shows the recovered
endpoint is a property of the **intervention (LR-schedule re-plasticization)**,
not of model scale. The Week-4 finding was that lr_shift pins the endpoint at
~0.77 *regardless of how deep the null was* (endpoint-invariant to deficit depth
within 23M). A ~0.767 endpoint at 60M extends that same invariance **across
scale** — the mechanism produces a fixed absolute recovery independent of size,
even though "% of gap" or "% of ceiling" recovered looks *worse* at 60M precisely
because the ceiling is higher. Scale-invariance of the mechanism is a deeper
claim than clearing a percentage bar at one size.

**Discipline — the "same endpoint" test, as a number** *(fixed 2026-07-23,
before any run; replaces the earlier "mean±SD intervals overlap" wording, which
was a soft phrase doing a hard threshold's job — interval overlap is not a test,
it moves with whichever arm happens to be noisier).*

**THE GATE = a two-sided tolerance band on the 60M n=3 mean:**

> **SAME-ENDPOINT BAND: 0.767 ± 0.045 → [0.722, 0.812]**, applied to the
> **mean** of the three 60M lr_shift step-5000 endpoints.
> **Precision guard:** the 60M lr_shift SD must be **≤ 0.045** (the band's own
> half-width) — an arm cannot claim a tolerance tighter than its own scatter.

*Derivation of ±0.045 (not invented for this run; two routes, see the
independence caveat below):*
1. **Empirical.** The only pair of arms Week 4 concluded reach the *same*
   endpoint is 23M lr_shift (0.752/0.778/0.772) and cram+lr_shift
   (0.746/0.794/0.804). Their pooled 6-value scatter is SD = **0.0227**; 2 SD =
   **0.0455**. That is, measured, how far apart two endpoints we already call
   "the same" actually sit.
2. **Structural.** Half of 2c's substantive-lift threshold (0.10 / 2 = 0.05) —
   two endpoints are the same if they differ by less than half the smallest
   difference this study calls substantive.
Both land at 0.045–0.05; the tighter value is adopted.

> **INDEPENDENCE CAVEAT (state this in the write-up; do not claim two
> confirmations).** These two routes are **not independent evidence**. Both trace
> back to the same 23M n=3 dataset: route 1 is that dataset's endpoint scatter
> directly, and route 2's 2c threshold (0.10) was itself calibrated against the
> same runs' ~0.190 lift. Their agreement is **internal consistency** — it shows
> the band is coherent with the thresholds already in use elsewhere in this
> pre-registration, and that the number was not reverse-engineered to make a
> particular 60M result pass. It is **not** independent corroboration that ±0.045
> is the right tolerance, and it carries no external validation: a systematically
> different noise regime at 60M would not be caught by either route. The band's
> real warrant is the calibration check (it correctly classifies the two 23M arms
> we already treat as the same endpoint) plus its pre-registration before any 60M
> data exists — not a claim of two-source support.

*Calibration checks:* 23M lr_shift SD 0.0136 ✓ and cram+lr_shift SD 0.0310 ✓
both clear the precision guard; both arm means (0.767, 0.781) sit inside the
band ✓; all six individual 23M seed endpoints (0.746–0.804) sit inside it ✓.

**A formal equivalence test (TOST) is deliberately NOT the gate — verified,
not assumed.** Run on the lr_shift vs cram+lr_shift pair (the known-same
reference), TOST fails at every defensible margin: δ = 0.045 → p = 0.110,
δ = 0.05 → p = 0.086, δ = 0.06 → p = 0.054; it only passes at δ = 0.080, a
margin nearly half the size of the entire 0.190 recovery lift. n=3 vs n=3 has no
power for equivalence at a meaningful margin, so making TOST the gate would
guarantee that a true scale-invariance result is scored as a miss. TOST **is**
still computed and reported at δ = 0.05 as a supporting statistic, explicitly
flagged non-gating and underpowered, with this calibration failure cited.

**Reading the band (all three directions pre-registered):**
| 60M lr_shift n=3 mean | sub-case | verdict |
|---|---|---|
| inside [0.722, 0.812], SD ≤ 0.045, 2b fails | **(a)** | **SCALE-INVARIANT RECOVERY ENDPOINT** — feature prominently, as argued above. |
| **< 0.722** | **(b)** | Ordinary weaker partial recovery — "recovery is weaker at 60M." NOT scale-invariance; must not be sold as such. |
| **> 0.812** | **(c)** *(new — the band is two-sided)* | **Endpoint scales with capacity** — recovery is *stronger* at 60M, a positive finding but the **opposite** of scale-invariance. Report as such; the Week-4 "pins ~0.77 regardless" claim would then be size-local, and must be restated that way. |
| inside band but SD > 0.045 | — | "Compatible with the 23M endpoint but imprecise at n=3" — do NOT feature; escalate to n=5 before any invariance claim. |

*Mechanical consistency of sub-case (a):* 2b failing with an in-band endpoint
requires 0.90 × d0_60M > 0.722, i.e. **d0_60M > 0.802** — 60M's anchor must
genuinely exceed 23M's 0.791 for this cell to be reachable at all, which is
exactly the "higher ceiling raises the bar" premise. If d0_60M ≤ 0.802 the cell
is unreachable and an in-band endpoint simply passes 2b (full threshold
generalization).

If 60M instead passes all of 2a/2b/2c, that is full threshold generalization —
also a strong result, reported plainly. The special callout is reserved for the
specific, easily-under-sold (a) pattern: same absolute endpoint, higher bar.

### CROSS-RULE COMBINATION, named before observation: **SHALLOW-CP FULL RECOVERY**
*Pre-committed 2026-07-23.* Rule 1b and the same-endpoint band were derived
independently of each other, so their composition needs a name **now**, not after
it appears. The combination in question is coherent and plausible:

> a size lands **ATTENUATED CRITICAL PERIOD** (Rule 1b: 0.05 ≤ gap < 0.15,
> Welch p < 0.05 — a real but shallower CP) **and** its lr_shift n=3 mean falls
> **inside the same-endpoint band [0.722, 0.812]** with SD ≤ 0.045 — i.e. a
> shallower deficit that is nonetheless carried all the way to the usual ~0.767
> endpoint.

**Name: SHALLOW-CP FULL RECOVERY (deficit-independent endpoint).**

**Claim tier — corroborative, NOT a generalization claim.** 2a (gap ≥ 0.15) fails
by construction here, so per Rule 2 this size makes **no** reopening claim of
either tier: not *threshold reopening generalizes*, not *the effect generalizes*.
The lr_shift arm remains formally non-confirmatory. What the cell earns is a
named **secondary/observational** finding, reported in its own right and
explicitly labelled as such.

**Why it is worth naming rather than discarding.** It is the cleanest available
instance of the Week-4 endpoint-invariance mechanism: lr_shift pinned ~0.767 at
23M *regardless of how deep the null was* (0.524–0.626). A shallow-CP size
reaching the same band extends that invariance to a **deficit depth outside 23M's
observed range** — the endpoint tracks the intervention, not the size of the hole
it is filling. That is the same argument as sub-case (a), on the severity axis
instead of the scale axis. Report it beside, and framed as continuous with, the
23M "pins ~0.77 regardless of null depth" result.

**Required reporting** (so the tier is unmistakable): per-seed d0, d50, and
lr_shift side by side; the gap with its Welch p; the endpoint mean ± SD against
the band; and an explicit sentence that 2a is not met and no generalization
claim is being made.

**Overshoot sub-clause.** This combination can mechanically place the endpoint
**above** the size's own d0 (e.g. d0 = 0.70, gap = 0.10 → d50 = 0.60, endpoint
0.767 > d0). If `lr_shift_mean > d0_mean` at any size, flag it as **OVERSHOOT**
and report it explicitly: the treated model exceeds its own early-exposure
anchor, which breaks the "recovery *toward* a baseline" framing entirely and
means 2b (≥ 0.90 × d0) is passing for a reason the rule was never designed to
capture. An overshoot is reported as an anomaly requiring its own explanation —
never silently absorbed into a recovery percentage or a 2b pass.

### ADDITION 2026-07-23 (post-probe): forgetting DRIFT CONTROL at 60M
The seed-1 probe passed Rule 1 at 60M but **breached the forgetting gate**
(c − b = +1.790 > +1.0) while recovering 99.7% of its own anchor. Added to the
60M fill, **without changing the gate or any threshold**: a **no-op fork**
control per seed — fork d50@2500 and replay the schedule segment starting at
2501. `ShiftedLRWindow` maps LR to `schedule(step − at_step + source_start)`, so
`source_start == at_step == 2501` reproduces `schedule(step)` exactly: identical
fork machinery, identical checkpoint and data stream, unmodified LR.
Run id: `arith_delay50_exp20_size60M_seed{s}_ivlrshift2501at2501`.

It separates three explanations of the breach:
- **(a) real cost** — lr_shift genuinely costs more general capability at scale.
- **(b) gate miscalibration** — the absolute +1.0 was justified at 23M as "≈38%
  of the 2.64 delay cost"; at 60M the delay cost b − a is **+0.016**, so that
  justification does not transfer. (The breach does survive the *relative*
  reading — 4.34% vs the gate's ~1.97% — so this alone does not explain it.)
- **(c) horizon artifact** *(added from probe data already on disk)* — the cost
  is transient re-convergence from the replayed warmup, measured before it
  finishes. Over [3750, 5000] the 60M lr_shift arm falls 47.806 → 43.027
  (**−4.779**) while the null falls 42.057 → 41.237 (**−0.820**): the arm is
  still descending **5.8× faster than the null at the horizon**. At 23M the arm
  was likewise still descending (−0.93 vs −0.19 per 200 steps) but had already
  crossed *below* the null, ending at −0.976. Whether c − b reads positive or
  negative may therefore depend largely on how far re-convergence has run by a
  fixed 5000-step horizon.
  **A decisive test of (c) needs a horizon extension past 5000, which is NOT
  wired on the fork path (no `--total-steps` override). Flagged, not built.**

**Taxonomy deliberately NOT decided.** The "recovers fully but breaches the
forgetting gate" cell is not covered by the four-cell 2b×2c matrix (which
assumes the gate passes). Naming it is **deferred until the n=3 fill and the
drift control return** — naming a category off one seed is exactly the
premature-labeling error caught twice this week (the syntax seed-1 baseline
artifact, and the 11M capacity over-claim).

Forgetting is checked independently at every size: intervention `val_ppl` cost
(c − b) ≤ +1.0, same as 23M; a recovery that breaches it is forgetting, not
reopening (CLAUDE.md operational definition), regardless of 2a–2c.

## Self-consistency check — all thresholds PASS at 23M (they must; 23M is the positive)
- 2a: gap 0.214 ≥ 0.15 ✓  (min-seed 0.166 ≥ 0.15 ✓, barely — by design)
- 2b: lr_shift 0.767 ≥ 0.90 × 0.791 = 0.712 ✓
- 2c: lift 0.767 − 0.577 = 0.190 ≥ 0.10 ✓
- Rule 1b: gap 0.214 ≥ 0.15, Welch one-sided p = 0.0085 → **CP confirmed** ✓ (not attenuated, not inconclusive)
- Same-endpoint band: 23M's own arms sit inside [0.722, 0.812] (0.767, 0.781) and clear the SD ≤ 0.045 guard ✓
None of these thresholds retroactively disqualifies the established 23M finding.

## Reporting rule (per CLAUDE.md standing rules)
Report **absolute endpoints ± SD** at each size (d0, d50, lr_shift shown as three
numbers side by side), the per-seed Detect gap and recovery lift, and the 3-size
Detect-gap and lr_shift curves. Any percent-of-gap figure is seed-specific and
secondary. The headline is the absolute d0/d50/lr_shift triads at 11M/23M/60M and
whether Rules 1–2 pass at each — NOT a single recovery percentage.

## 60M and the architecture ceiling — deliberate, not drift
60M exceeds the locked 10–30M spec. The syntax sweep already carved this
exception (VRAM-verified 2.95 GB peak on the 8 GB 4060); extending it to
arithmetic for the same scale-robustness purpose is a **deliberate** decision
recorded here, matching the syntax sweep's sizes so the cross-skill comparison is
apples-to-apples. It does not reopen the ceiling for the main grid.

## Read-out (to script at n=3, not eyeball)
Pull step-5000 `arith_accuracy` and `val_ppl` from each run's `metrics.csv`,
assemble the per-size d0/d50/lr_shift ± SD table, compute gap and lift, and apply
Rules 1, **1b** and 2 programmatically. (A `scripts/arith_scale_gate.py` mirroring
`scripts/szsweep_gate.py` should be written at read-out time so the verdict is
mechanical.) It must emit, per size: the Rule-1 probe branch; the Rule-1b named
Detect verdict incl. the Welch one-sided p; the 2a/2b/2c booleans and four-cell
verdict; and for 60M the same-endpoint band membership (mean, SD, sub-case
a/b/c) plus the non-gating TOST at δ = 0.05.
