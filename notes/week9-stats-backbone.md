# Week 9 — statistical backbone and whole-paper evidential ledger

**Purpose.** Fix, once, (a) which statistic carries each finding-cluster in the write-up,
(b) what evidential-strength label each claim gets, and (c) which artifact owns each
number. Everything below is either already computed or computed by
`scripts/week9_stats_backbone.py` → `logs/week9_stats/backbone_stats.{json,md}`.

This is a *decision* document, not a new investigation. No new runs are implied by it.

**What changed relative to the Week-9 plan in CLAUDE.md:** all three pre-registered
analysis designs are retired as un-estimable on the surviving design, and replaced by
tests the data can actually support (§2.1, §2.2, §2.4). The retirements are stated with
reasons so the write-up can say what was planned and why it was not run — that is a
methods-section sentence, not a gap.

---

## 1. Label taxonomy (three labels, two annotations, one non-bucket)

Every claim in the paper carries **exactly one** label, fixed in §3, and the same label
in the abstract, its section, and the discussion. The label determines the strongest
permitted verb.

| Label | Criteria (all must hold) | Permitted verbs |
|:--|:--|:--|
| **ESTABLISHED** | Replicated at n ≥ 3 (or n ≥ 24 for rate claims); direction stable in **every** replicate; decided by a rule fixed **before** the data (gate, pre-registration); no unresolved confound | "shows", "establishes", "confirms" |
| **SUGGESTIVE** | Direction consistent, but ≥ 1 of: n < 3 · single method · single mixture · observational rather than manipulated factor · significance estimator-dependent (0.01 < p < 0.10 under a robust SE) | "suggests", "is consistent with", "points toward" |
| **BOUNDED-NOT-EXCLUDED** | A null/attenuated result whose interval still contains the effect it failed to confirm, **or** a cost bounded above but not point-estimated. Must be quoted **with its bound** | "does not survive replication", "is bounded above by X", "is not excluded" |

Two **annotations** attach to a label rather than replacing it:
`(single-method)` — only one probe discriminates, *demonstrated* not assumed (§2.4);
`(single-mixture)` / `(single-size)` — scope limit on generalisation.

And one **non-bucket**: **NOT-A-CLAIM** (§8) — computed quantities that appear in the
paper for completeness but support no assertion. A reader who recomputes them must find
them already discussed, or they read as a suppressed result.

---

## 2. Cluster backbones

### 2.1 C1 — Detect: does the arithmetic critical period exist, and does it scale?

**RETIRED:** `final_accuracy ~ delay × model_size` (mixed-effects). Not estimable here.
After the pre-registered Rule-1 FLOOR excludes 11M, the design is 2 delays × 2 sizes ×
3 seeds = **12 observations with exactly one observation per seed × cell**. There is no
within-cell replication for a random effect to pool over, variance components on three
seeds are unidentifiable, and `delay` has only two levels at n = 3 (d25/d55/d75 exist at
23M seed 1 only). A random-intercept model would reproduce the paired difference with
worse small-sample calibration and a false impression of pooling.

**BACKBONE:** seeds are *crossed* (same seed = same init at both delays), so the
design-appropriate statistic is the **per-seed paired deficit**, tested per size, Holm-
corrected across the two confirmatory sizes (family F1, §4). Welch is reported beside it
because the pre-registered gate used Welch.

| size | early anchor | delayed null | deficit (mean ± SD) | paired t (df 2) | p | Holm | Hedges g |
|:--|:--|:--|:--|--:|--:|--:|--:|
| 23M | 0.791 ± 0.007 | 0.577 ± 0.051 | **0.214 ± 0.055** | 6.74 | 0.0107 | 0.0107 | 2.22 |
| 60M | 0.829 ± 0.027 | 0.600 ± 0.040 | **0.229 ± 0.037** | 10.64 | 0.0044 | 0.0087 | 3.51 |
| 11M | 0.538 ± 0.018 | 0.357 ± 0.070 | 0.181 ± 0.087 | 3.60 | 0.0345 | *excluded* | 1.19 |

Always print the per-seed values with the table; the delayed null is the seed-variable
side (SD 0.051 vs the anchor's 0.007) and that is itself a finding.

**Size moderation** (is the deficit *larger* at 60M?): Welch p = 0.72, **MDE at 80 %
power = 0.144**. Report the MDE inline. "Scale-robust" means *the critical period is
present and of similar magnitude at both sizes*; it must never be written as "size does
not matter", which this design cannot support against a deficit of ~0.21.

**This is the paper's WELL-POWERED size null; C4's (§2.4) is the underpowered one.
Write them as a matched pair, never in isolation.** They are structurally identical —
same n = 3 per group, same Welch test, same two sizes, both non-significant — and they
carry *opposite* evidential weight, for one reason only: the residual SD. C1's MDE
(0.144) sits **below** the deficit it is asked about (0.21), so its null genuinely
constrains — a size effect big enough to matter would have shown. C4's MDE (0.809)
**exceeds the entire 0.681 span** its outcome occupies, so its null constrains nothing.
Presenting the two together converts a methods caveat into an argument: it demonstrates
that the paper reads non-significance off the MDE rather than off the p, and it is the
cleanest available illustration of why "n = 3, p > 0.05" is not one verdict but two.
A reader who sees only C1 will over-trust C4; one who sees only C4 will under-trust C1.

**11M is NOT-A-CLAIM but must be reported.** Its paired deficit is nominally significant
(p = 0.0345, g = 1.19) — a reader recomputing it will find that. It is excluded by the
*pre-registered* FLOOR rule because the early anchor itself (0.538) never plateaued and
sits below the 0.60 floor: the comparison is between two under-trained models, so the
gap is not interpretable as a timing effect. Report the number, the exclusion, and the
reason, in that order.

**Label:** arithmetic critical period at 23M and 60M — **ESTABLISHED**.
Scale-robustness of its *magnitude* — **ESTABLISHED** (as "similar at both sizes",
with the MDE quoted).

### 2.2 C2 — Reopen: does the LR-shift intervention reopen it, and at what cost?

**RETIRED:** `recovery_percentage ~ intervention_type × skill × delay_onset`. Three
independent reasons: (i) the recovery-% denominator is the *seed-variable* null, which
the standing rule already bans as a headline; (ii) `skill` has one surviving level
(syntax LD dropped) and `delay_onset` one level at n ≥ 3 (d50) — two of three factors
are constants; (iii) `intervention_type` has n = 3 for exactly one level (lr_shift) and
n = 1 for the other five, so no interaction is identifiable.

**BACKBONE:** three separate things, none of them a factorial model.

**(a) The pre-registered gate, on absolute endpoints.** Both endpoints shown, never
collapsed into a ratio:

| size | treated (lr_shift) | delayed null | early anchor | gate 0.90×own d0 | margin | lift over own null, p |
|:--|:--|:--|:--|--:|--:|--:|
| 23M | **0.767 ± 0.014** | 0.577 ± 0.051 | 0.791 ± 0.007 | 0.712 | +0.055 | 0.0065 |
| 60M | **0.796 ± 0.022** | 0.600 ± 0.040 | 0.829 ± 0.027 | 0.746 | +0.050 | 0.0027 |
| 11M | 0.747 ± 0.021 | 0.357 ± 0.070 | 0.538 ± 0.018 | 0.484 | +0.262 | *degenerate, see below* |

Both confirmatory endpoints fall inside the pre-registered same-endpoint band
[0.722, 0.812] → the *recovered endpoint is scale-invariant*, which is a stronger and
more surprising statement than the recovery itself and should be foregrounded.

Report alongside it the **variance collapse**: the null's cross-seed SD (0.051 at 23M,
0.040 at 60M) drops to the treated arm's (0.014, 0.022). The intervention makes the
outcome *more reproducible*, not just higher.

**11M's gate PASS is degenerate and is NOT-A-CLAIM.** The gate is 0.90 × an anchor that
failed the floor, and the arm (0.747) *exceeds its own early anchor* (0.538) — evidence
that the 11M anchor was LR-starved, not that reopening over-recovers. The 11M arm enters
the paper only as the H2/H3 mechanism decomposition: delayed lift +0.389 = generic
+0.115 + exposure-alignment +0.274, i.e. **~70 % of the effect requires the delayed
exposure profile** (SUGGESTIVE, exploratory).

**(b) Arms scored per seed against their OWN fork source. EXTENDED TO n = 3 on 2026-08-13
— the label survives, its basis does not.** The n = 1 version of this block scored each arm
against the null's cross-seed *range* because one seed landing inside a 0.10-wide range is
indistinguishable from the null. At n = 3 the arms are scored by **per-seed sign consistency**
against each arm's own fork source (0.571 ± 0.061, range 0.506–0.626 — fork-source lineage,
deliberately not the Detect null; do not reconcile). Four arms now fail on *disagreeing
signs* rather than on insufficient n. The verdict wording is unchanged and "fails to reopen"
stays banned; only the reason moves.

| arm (23M, n = 3) | endpoint | per-seed lift | verdict |
|:--|--:|:--|:--|
| reset (embedding) | 0.585 ± 0.001 | +0.080 / −0.040 / +0.002 | no consistent direction |
| reset + LR spike | 0.571 ± 0.042 | +0.070 / −0.100 / +0.028 | no consistent direction |
| LR spike alone | 0.578 ± 0.048 | +0.030 / +0.004 / −0.014 | no consistent direction |
| cram | 0.577 ± 0.005 | +0.068 / −0.052 / 0.000 | no consistent direction |
| **freeze** | **0.422 ± 0.081** | −0.174 / −0.182 / −0.092 | **harmful** — below the *entire* null range at every seed |

Two endpoints moved materially against their published n = 1 values (LR spike 0.536 → 0.578,
freeze 0.332 → 0.422), and seed 1 gave the largest lift of any seed for all four
non-directional arms. Any surviving "(n = 1)" qualifier on these arms is now stale and must
be struck wherever it appears.

This is the one place where the write-up can currently say something directional from
n = 1, and only for freeze. Prediction #2 (reset predicted best) is **not borne out**;
say that as "reset showed no detectable recovery at n = 1 and flagged forgetting", not
as "reset fails".

**(c) Forgetting: always the three-number decomposition** (early a, delayed null b,
treated c), never a single delta. 23M: no breach. 60M: c − b = **+1.670 ± 0.104 ppl**,
tripping the pre-registered +1.0 gate in all three seeds — reported as a trip, honored
as written. Against it: the no-op fork drift control (≈ 0 on both ppl and accuracy) rules
out fork mechanics, and at matched LR the treated arm *leads* the null by
**+2.477 ± 0.240 ppl**, a lead exceeding the breach, because the arm is read mid-anneal
at 6× the null's terminal LR.

**(d) The generic-booster discriminator, at a confirmatory size (added 2026-08-10).**
The obvious alternative reading of the schedule account is that `lr_shift` is just a
better optimizer setting, in which case it would lift *any* run. Applying the identical
shift to the **early** (d0) 23M runs, forked from their own step-2500 checkpoints:

| arm (23M, n = 3) | endpoint | lift over own null |
|:--|--:|--:|
| d0 anchor | 0.791 ± 0.007 | — |
| **d0 + lr_shift (discriminator)** | **0.783 ± 0.025** | **−0.008** (per-seed −0.040 / +0.016 / 0.000) |
| d50 null | 0.577 ± 0.051 | — |
| d50 + lr_shift | 0.767 ± 0.014 | +0.190 |

Alignment component +0.198, positive at every seed, paired *t*(2) = 5.17, one-sided
*p* = 0.018. This **supersedes claim 8's 11M-only basis** (which the Rule-1 FLOOR
excludes). The 11M generic component was +0.115 against 23M's −0.008, consistent with
the 11M anchor being LR-starved — the same reading its degenerate gate already forced.

**Scope discipline:** this rules out ONE alternative reading. It is *not* a second
method corroborating claim 7 — it tests a different contrast (does the shift help an
undelayed model?), not the same contrast by another route. Do not write it as
cross-method validation; the §10 rule has already cost the project twice.

**Label:** lr_shift reopening at 23M and 60M — **ESTABLISHED**; scale-invariant
endpoint — **ESTABLISHED**; the other four Tier-1 interventions — **SUGGESTIVE (n = 3,
signs disagree)**, with freeze's harm the only directional one and now directional at three
seeds rather than one; 60M general-LM cost —
**BOUNDED-NOT-EXCLUDED**, residual bounded above by 1.67 ppl, no point estimate quoted
(the anneal slope is unmeasured; the arm-extension is the only run that would collapse
it, and is future work).

### 2.3 C3 — the syntax long-distance null

**BACKBONE: an accumulation of nulls is only interpretable if each probe is shown able
to detect *something*.** Every row below carries either a positive control or a power
statement; that pairing is the argument, not the count of nulls.

| # | probe | result | why it is interpretable |
|--:|:--|:--|:--|
| 1 | endpoint by delay (n = 3) | no gap; d0 seed-variance exceeds the whole effect | the same harness resolves the arithmetic gap (0.214) on identical machinery |
| 2 | acquisition dynamics | Welch p = 0.46 | — (underpowered alone; carried by #3) |
| 3 | rate / bimodality | pre-registered compound gate over **72 runs** (24 seeds × 3 sizes): DISCONFIRM — fails FLOOR@0.56 (top size 0.208 < 0.40) and TREND@0.57 (p = 0.074); passes only at the permissive 0.55 | the powered test, sized in advance by simulation; supersedes the n = 8 rate estimate (which needed 18–82 seeds) |
| 4 | representational probe | d0 vs d50 **p = 0.48** *while* the probe detects structure at selectivity 0.60, **p < 1e-8** | **positive control passes** — the strongest null in the set: structure is encoded, its timing-dependence is not |
| 5 | LR-budget (hilr50, n = 3) | ΔLD +0.011 ± 0.005 | Δlocal > ΔLD in **every** seed and val_ppl improves → a non-specific whole-LM lift, not an LD rescue |

**Presentation rule:** these are five *different hypotheses*, not five tests of one. Do
not write "confirmed five ways" as if it were 5× the evidence; write "five distinct
routes to the same negative, each with its own detectable-effect floor".

**Label:** **ESTABLISHED (null)**, scoped to ≤ 60.3M parameters, the ~10M-word BabyLM
budget, and the continuation-scoring licensing-interaction metric. Outside that scope it
says nothing.

#### 2.3.1 Claim 11b — the arithmetic/syntax dissociation (added 2026-08-03)

This conjoins claims 1 and 11 and was **absent from the ledger until now**, despite the
master plan naming "a clean dissociation" as hit-criterion #1 (see the §10 standing rule
this omission produced). Note the **form differs from the plan's**: the plan's worked
example was *reopening*-side ("arithmetic is reopenable, syntax isn't"); what the data
support is *detection*-side. Syntax never showed a deficit, so no syntax reopening was
ever attempted — do not write the plan's version.

**The headroom asymmetry is the whole scoping problem.** State it as a table, not a
caveat sentence:

| | arithmetic | syntax LD |
|:--|--:|--:|
| early-exposure anchor | 0.791 ± 0.007 | 0.528 (24 seeds) |
| chance floor | ~0 (exact match) | 0.500 (by metric construction) |
| **headroom** | **0.791** | **0.0283** |
| observed deficit | 0.214 (**27.1 %** of headroom) | none detected |
| proportionally-matched deficit would be | — | 0.0077 (**0.49** delay-0 seed-SD) |
| MDE @ 80 % power, n = 3/group | — | **0.0482** |

**The LD design's MDE exceeds its entire headroom by 1.7×.** A *total* collapse of the
skill to chance (0.0283) sits below the smallest effect the design could resolve, and a
susceptibility proportionally identical to arithmetic's (0.0077) would need **n ≈ 67 per
group**. So the behavioural null cannot, on its own, distinguish *syntax is less
susceptible* from *syntax had no room to show susceptibility*. This is C4's failure mode
recurring — read power off the MDE, never off the p.

**What does and does not relieve it:**
- **Does not — the LR-budget null (probe 5).** It establishes that the ceiling is not an
  optimization-budget artifact (hilr50 lifted LD +0.017 ≈ 1 seed-SD, but lifted *local*
  +0.021 and improved val_ppl → non-specific). That makes the ceiling *real and immovable
  by that lever*, which **preserves** the dynamic-range limit rather than relieving it.
  Do not cite it as evidence the floor isn't masking a critical period; it is evidence
  about the floor's origin, not about what the floor conceals.
- **Does — the representational probe (probe 4).** It is the one readout **not bounded by
  the 0.0283 headroom**: it carries genuine dynamic range (selectivity 0.60, p < 1e-8)
  and is flat across timing (d0 vs d50, p = 0.48). If delayed exposure damaged the
  underlying representation, this probe had the range to register it and did not. **11b
  is load-bearing on probe 4, not on the accumulation of behavioural nulls.**

#### 2.3.1a The susceptibility question under the RATE readout (checked 2026-08-04)

The headroom argument above is a property of the **mean-endpoint** readout. It does
**not** transfer to the rate readout, and that had to be checked separately rather than
inherited — the μ_low + 3σ_low / P(high) machinery built for the size sweep is a
different statistic, and a rate can collapse (0.25 → 0) while the mean barely moves.

**Status: UNTESTED, not unmeasurable.** The rate machinery was only ever pointed at the
d0-only *reposition* question. The seed inventory is asymmetric — **24 delay-0 seeds per
size** (szsweep, cached item arrays) but only **3 delayed seeds, at 23M alone**. The
d0-vs-d50 rate contrast was never run.

d0 exceedance rates, computed from the cached arrays (`logs/szsweep_eval/per_seed_ld.json`):

| size | mean | P(high) @ 0.55 | @ 0.5593 (derived) | @ 0.57 |
|:--|--:|--:|--:|--:|
| 11M | 0.5283 | 1/24 = 0.042 | 0/24 = 0.000 | 0/24 |
| 23M | 0.5384 | 5/24 = 0.208 | 2/24 = 0.083 | 1/24 |
| 60M | 0.5518 | 12/24 = 0.500 | 6/24 = 0.250 | 2/24 |

Delayed seeds needed to detect a *collapse* of the high mode (observe 0/n, one-sided
Fisher exact vs the observed d0 counts, α = 0.05):

| test | needed | with the n = 3 we have |
|:--|--:|--:|
| 23M @ 0.5593 | **82** | p = 0.786 |
| 23M @ 0.55 | 18 | p = 0.526 |
| **60M @ 0.5593** | **14** | *no delayed 60M seeds exist* |
| 60M @ 0.55 | **6** | — |

**Two consequences.** (i) At 23M — the only size where the delayed contrast was run — the
rate readout is *worse* powered than the mean (82 vs ~67), so nothing is recoverable
there. (ii) At **60M it is cheap: 14 seeds** (6 at the permissive threshold), because the
high mode is four times more common. Test feasibility *improves with scale*, tracking the
same monotone trend the size sweep found. Hedge that stays: these n's assume delay
*eliminates* the mode; if it merely halves it, 59 per group at 60M and 203 at 23M.

**Label: ESTABLISHED (scoped).** Not plain ESTABLISHED. The scope is not decoration —
the *absolute* dissociation is established (one skill loses 0.214, the other loses
nothing measurable, under matched design, matched total exposure and a shared pipeline).
The *differential-susceptibility* reading is **not established, and the reason is
readout-specific**: unmeasurable at the achieved n under the mean-endpoint readout
(needs ~67/group against 0.028 of headroom), and simply **un-run** under the rate
readout, where it is tractable at 60M. Write the absolute form in the abstract.

**How to write the susceptibility reading — hedged EMPIRICAL, not pure hypothesis.**
Do not write "we cannot know whether syntax is less susceptible". Write that the
question is well-posed, that the design which answers it is specified, and that it was
not run: *a rate-collapse test at 60M (14 delayed seeds against the existing 24-seed d0
baseline) would settle it; at 23M the same test needs 82 and is out of reach.* Naming
the price is a stronger and more honest position than declaring the question closed.

**Residual limitation for the discussion (state it, do not bury it):** syntax's low
acquisition ceiling remains a partial alternative explanation for the null that this
design cannot fully exclude. Probe 4 constrains it; the 60M rate test would settle it.

### 2.4 C4 — circuit identity: repair vs reorganization

**RETIRED:** `circuit_overlap ~ intervention_type × skill`. One intervention (lr_shift)
and one skill (arithmetic) survive; both factors are constants.

**EVAL-CONTAMINATION CORRECTION (2026-07-28) — all C4 numbers below are the corrected
ones.** `week8_circuit_identity.py` built its eval set with `ArithConfig(seed=<run
seed>)`, but only the seed-1 draw is held out of training: the seed-2/3 draws contain
**19.4 % / 17.4 % trained-on problems**, at both sizes. All six triads were re-scored on
the held-out draw, with the seed-1 triads reproducing exactly as a no-op control. Full
read-out: `logs/week9_stats/cleaneval_recheck.md`. The correction changed the primary
statistic materially, so the superseded values are shown beside it everywhere.

**BACKBONE:** reciprocal mean-ablation transfer over six triads, deficit depth as the
ordering variable.

- **Primary:** Spearman ρ = **−0.714**, **exact permutation p = 0.0681** (n = 6;
  asymptotic 0.111). *Superseded (contaminated): ρ = −0.943, p = 0.0083.* Use the exact
  p: with only 720 orderings it is computable, and quoting an approximation when the
  exact is cheap is something this project has already had to correct twice.
- **Robustness:** leave-one-out is **fragile** — 4 of 6 single-point deletions push p
  above 0.05 (range ρ = −0.90 … −0.50). The relation is carried by the whole six-point
  set, not robust within it. *Superseded: robust, worst p = 0.042.*
- **Out-of-sample:** the 23M-only fit no longer supports anything (r = −0.85, p = 0.351;
  residuals against the 60M points −0.30 / −0.15 / −0.20, all one-sided). **Dropped as
  evidence.**
- **Size — the collinearity argument reverses, in the paper's favour.** On corrected
  numbers the deficit ranges *overlap* and the means are nearly identical (23M 0.228,
  range 0.182–0.292; 60M 0.229, range 0.202–0.260), so size is **separable** from
  deficit depth — the question the earlier read-out called unidentifiable is now
  **estimable**. It is *not* thereby **answered**: 23M 0.571 ± 0.349 vs 60M
  0.350 ± 0.131, Welch p = 0.39, but **MDE at 80 % power = 0.809** against an outcome
  whose six triads span only 0.187–0.868 (span 0.681), 95 % CI on the difference
  [−0.537, +0.979]. The design could not have detected *any* difference realizable
  within the measured range, so this null excludes nothing — it is uninformative, not
  evidence of no effect. Contrast C1, where MDE 0.144 sits below the 0.21 deficit and
  the null therefore constrains something. What remains directional is **reproducibility**:
  SD 0.349 vs 0.131 (2.7×, down from 5.1×) — not inferentially useful at n = 3 per group.
#### 2.4a The categorical classifier's *repair* arm — provenance, and why claim 13 is now stated as a continuum (2026-08-05)

**Question asked:** were the verdict thresholds pre-registered before data, or set on an
unverified expectation? **Answer: neither cleanly — and the distinction matters more than
the rewording it prompted.**

- The Week-8 pre-registration (`notes/week8-diagnosis-seed-scope.md`, 2026-07-21, written
  *before the tooling existed*) fixes **seed scope and which diagnostics run** — how many
  seeds, which battery, why seed 2 before seed 3. It contains **no verdict thresholds at
  all**, and no Jaccard criterion.
- The thresholds are a **code-level default** in `scripts/week8_circuit_identity.py:276-278`:
  `REPAIR` iff `transfer ≥ 0.6 AND jaccard ≥ 0.4`; `NEW CIRCUIT` iff
  `transfer ≤ 0.3 AND control ≥ 2.0`. They were written as part of the tool, not registered
  in advance of the data. The 0.4 encodes an **unverified expectation** that a repaired
  circuit reuses ≥ 40 % of the top-*k* head set.
- The rule **contradicts its own docstring**, asymmetrically. The comment immediately above
  it reads "Primary = transfer_ratio … **Jaccard of the head sets corroborates**" — yet the
  code makes Jaccard a hard conjunct on the *repair* branch only, while the reorganization
  branch uses the control ratio and never consults Jaccard. Only one of the two verdicts
  can be blocked by the corroborative measure.
- It was **empirically contradicted at the first opportunity and formally demoted**:
  `notes/week8-60M-seed2-prereg.md` (2026-07-26) states "Jaccard is corroborative only (it
  did NOT discriminate at 23M: seed 2 had Jaccard 0.09 *with* transfer 0.71)". So by
  2026-07-26 the project had recorded, in a pre-registration, that Jaccard must not gate —
  **and the classifier was never updated.** Max Jaccard across all six triads is 0.20, so
  the repair arm was unsatisfiable from the first triad onward.

**Consequence.** "No triad was classified as repair" is an artifact of a stale conjunct the
project itself had already superseded in writing, not a substantive finding about the
models. It would be equally wrong to *use* the superseding rule to relabel 23M seed 3 as
repair after seeing its number — that is reading a verdict opportunistically. The way out
that assumes nothing is to **stop reporting a categorical verdict and report the continuous
axis**, which is the primary readout under both the code's docstring and the 07-26
pre-registration. Claim 13 is reworded accordingly; the underlying measurements are
untouched. The verdict strings stay in the released logs, and §5 shows the verdict column
so the absence of a repair row is visible rather than quietly dropped.

**Standing lesson (companion to §7's "audit the replacement column").** A decision recorded
in a pre-registration does not propagate itself into code. The demotion of Jaccard was
written down correctly, in the right document, and then had no effect for ten days because
nothing re-ran the classifier or edited its rule. When a pre-registration supersedes an
operational threshold, grep the implementation in the same sitting.

- **The seed-2 vs seed-3 inversion survives the correction** and slightly widens: on the
  common held-out draw, deficits 0.182 vs 0.210 with transfers 0.659 [0.611, 0.708] and
  0.868 [0.832, 0.903] — CIs still disjoint, deeper seed still showing *more* repair.

**SINGLE-METHOD, by demonstration.** State it in that exact form, with the evidence:
CKA sits at 0.66–0.82 for every triad regardless of verdict; single-head patching is
flat (r = 0.045, p = 0.93 against the primary readout across all six triads); pruning's
answer *changes sign with λ* (r = −0.94 → +0.12 → +0.87) and the λ that agrees is the
one that over-prunes; whole-layer ablation saturates. Three methods checked and shown
non-discriminating is a *stronger* methods paragraph than one method presented alone —
write it as a result, not an apology.

**Labels:** "recovered models range from repair to reorganization" — **ESTABLISHED
(single-method)** (extremes 0.19 vs 0.87 with disjoint bootstrap CIs and control ratios
> 1 throughout); "60M reorganizes at every seed (0.350 ± 0.131, n = 3)" —
**ESTABLISHED (single-method)**; "deficit depth grades circuit identity" —
**BOUNDED-NOT-EXCLUDED** — *downgraded from SUGGESTIVE by the contamination correction*:
the direction persists (ρ = −0.71) but it is not significant at n = 6 (p = 0.068) and is
LOO-fragile, so it is neither supported nor excluded; "transfer level differs by size" —
**NOT-A-CLAIM**, estimable at matched deficit but underpowered (p = 0.39, MDE 0.809 >
the outcome's full 0.681 span) — report as untested, never as a tested null; "the
recovered circuit is redundant" — **NOT-A-CLAIM** (sufficiency untested: single-head
patching lacks dynamic range; joint head-group patching would be required).

**Two other Week-8 tools share the contaminated-draw pattern** (`week8_causal_trace.py`,
`week8_pruning_feasibility.py`). Neither supports a surviving claim — patching was
retracted as non-discriminating, pruning was a feasibility NO-GO — so no conclusion
depends on them; both must be re-run on the held-out draw before any future use. Say this
in the methods note rather than leaving it implicit.

### 2.5 C5 — subject–verb agreement attraction (bonus cluster)

**BACKBONE:** unit of analysis is *one trained model*. Slopes are OLS on log₁₀(params);
because the arithmetic mixture is imbalanced (3, 3, 8), **every slope in this cluster is
quoted as "p = OLS / robust"**, and the *paired within-model* (attraction − complexity)
contrast is the primary test for the dissociation, since both components are measured on
the same models. The unigram control (exactly +0.0000) is reported wherever attraction
is, as the discriminating-control.

| claim | statistic | label |
|:--|:--|:--|
| attraction errors are real and not frequency artifacts | +0.303, p = 8.6e-05; positive in 81/81 models across both mixtures; unigram control +0.0000 | **ESTABLISHED** |
| attraction declines with scale (syntax mixture) | −0.316/decade, p = 2.3e-17, n = 72 balanced | **ESTABLISHED** |
| …and in the arithmetic mixture | −0.207/decade, **p = 0.033 OLS / 0.073 robust**, n = 14; estimate stable (−0.21…−0.24) under every estimator | **SUGGESTIVE** |
| dissociation from complexity | syntax mixture paired −0.333, p = 3.8e-11 | **ESTABLISHED (single-mixture)** |
| …replicated in the arithmetic mixture | paired −0.095, CI [−0.314, +0.123], p = 0.36 | **BOUNDED-NOT-EXCLUDED** |
| 60.3M cross-mixture divergence | +0.065, CI [−0.013, +0.143], p = 0.091 at n = 8 (was +0.133 at n = 3) | **BOUNDED-NOT-EXCLUDED** — CI still contains +0.133; effects < ≈0.11 untested |
| below chance at the smallest size only | 11.5M 0.405; 60.3M arithmetic 0.626 ± 0.114, above chance p = 0.017 | **ESTABLISHED** (scoped per size) |
| trajectory effect | p = 0.053, power 53 %, MDE 0.125 (needs ~10 seeds) | **SUGGESTIVE** |

The English-exposure confound is **excluded** independently (exposure is constant across
sizes by design → predicts a uniform offset, interaction ≡ 0; ceiling |0.031| vs the
observed +0.133) — that is an exclusion and may be stated as one.

---

## 3. Whole-paper claim ledger

The single source of truth for labels. If a sentence in the draft asserts something not
in this table, either it is not a claim or the table is incomplete — resolve before
submission.

| # | claim (as it may be stated) | cluster | label |
|--:|:--|:--|:--|
| 1 | Delayed arithmetic exposure produces a persistent endpoint deficit at 23M and 60M | C1 | ESTABLISHED |
| 2 | The deficit is of similar magnitude at both sizes (MDE 0.144) | C1 | ESTABLISHED — **paired-null dependency (§5.5): may not ship without the cross-reference to C4's underpowered size null** |
| 3 | The deficit's *severity* is seed-variable, driven by the delayed endpoint | C1 | ESTABLISHED |
| 4 | An LR-shift intervention reopens it (0.767 ± 0.014 / 0.796 ± 0.022 vs nulls 0.577 ± 0.051 / 0.600 ± 0.040) | C2 | ESTABLISHED |
| 5 | The recovered endpoint is scale-invariant (both inside [0.722, 0.812]) | C2 | ESTABLISHED |
| 6 | The intervention collapses cross-seed variance in the endpoint | C2 | ESTABLISHED |
| 7 | Closure at d50 is largely LR-schedule, not representational entrenchment | C2 | ESTABLISHED (cram+lr_shift ≈ lr_shift ≫ cram, n = 3) |
| 8 | ~70 % of the 11M effect requires the delayed exposure profile (H3) | C2 | SUGGESTIVE (exploratory) |
| 8b | At 23M the reopening effect is **entirely** exposure-aligned: the identical shift lifts the early runs −0.008 (0.783 ± 0.025 vs anchor 0.791 ± 0.007) while lifting the delayed runs +0.190; alignment +0.198, positive at all 3 seeds, paired *t*(2) = 5.17, one-sided *p* = 0.018 | C2 | ESTABLISHED — but scoped as *rules out the generic-booster reading*, NOT as a second method confirming claim 7 (which rests on the cram contrast). See the §10 cross-method rule |
| 9 | Reset / reset+spike / spike / cram show no consistent direction; freeze is harmful | C2 | SUGGESTIVE (n = 3, per-seed signs disagree) — **re-based 2026-08-13**: label unchanged, basis moved from "one seed cannot separate it from the null" to "three seeds disagree in sign". Freeze's harm is now directional at 3 seeds (below the entire null range at every one). Endpoints restated at n = 3; the n = 1 figures for LR spike (0.536) and freeze (0.332) were seed-1 extremes |
| 10 | The 60M general-LM cost is bounded above by 1.67 ppl and is likely a horizon artifact | C2 | BOUNDED-NOT-EXCLUDED |
| 11 | Long-distance wh-licensing shows no detectable critical period under five distinct probes | C3 | ESTABLISHED (null, scoped) |
| 11b | Arithmetic and long-distance syntax dissociate under matched design: delayed exposure costs arithmetic 0.214 ± 0.055 and costs LD nothing measurable across five probes. **Scoped — syntax's acquisition ceiling (0.028 of headroom vs arithmetic's 0.791) is a partial alternative explanation that this design cannot fully exclude**, since the LD MDE (0.048) exceeds its total headroom. The absolute dissociation is established; the differential-*susceptibility* reading is not — unmeasurable at the achieved n under the mean readout, and **bounded but undecided** under the rate readout: the 60M rate-collapse test ran at 24 delayed vs 24 baseline seeds on 2026-08-13 (6/24 in both arms, Fisher p = 0.630) and excludes a *full* collapse of the high mode while leaving a *half* collapse open at ~59 seeds/group (§2.3.1, §2.3.1a, claim 24) | C1 × C3 | **ESTABLISHED (scoped)** |
| 12 | The structure is encoded but its readout is timing-invariant (probe positive control passes) | C3 | ESTABLISHED |
| 13 | Recovered models span a continuum from near-complete transfer of the original circuit (0.868) to full reorganization (0.187) | C4 | ESTABLISHED (single-method) — **reworded 2026-08-05; see §2.4a. Do not write "range from repair to reorganization": *repair* is a categorical verdict the classifier never awarded** |
| 14 | 60M reorganizes at every seed | C4 | ESTABLISHED (single-method) |
| 15 | Deficit depth grades circuit identity | C4 | **BOUNDED-NOT-EXCLUDED** (ρ = −0.71, p = 0.068, LOO-fragile) |
| 16 | 60M circuit identity is more reproducible than 23M (SD 2.7×) | C4 | SUGGESTIVE |
| 16b | Within the shallow-deficit regime, the deeper seed shows *more* repair (inversion) | C4 | ESTABLISHED (single-method; disjoint CIs on the common held-out draw) |
| 17 | Agreement attraction is real, not a frequency artifact | C5 | ESTABLISHED |
| 18 | Attraction declines with scale | C5 | ESTABLISHED (syntax) / SUGGESTIVE (arithmetic) |
| 19 | Attraction dissociates from general complexity | C5 | ESTABLISHED (single-mixture) + BOUNDED-NOT-EXCLUDED (replication) |
| 20 | Below-chance agreement occurs only at the smallest size | C5 | ESTABLISHED |
| 21 | Attraction declines *within* training (+0.392 → +0.303, steps 2500 → 5000, paired, n = 6 at 23.2M, 5/6 seeds) — weak evidence of unlearning | C5 | **SUGGESTIVE (exploratory)** — added 2026-08-05. p = 0.053 is marginal at 53 % power (MDE 0.125; ≈ 10 seeds needed). **The power numbers travel with the claim**; the mismatch cell's own improvement (+0.070, p = 0.041) is the part that is not marginal. Never a headline (already flagged exploratory in §4) |
| 22 | The d50 arithmetic deficit is **abolished under a WSD schedule** (−0.007 ± 0.013, per-seed +0.002/+0.000/−0.022) against 0.214 ± 0.055 under cosine, confirming claim 7's mechanism on a second, deliberately dissimilar schedule | C2 | **ESTABLISHED** — added 2026-08-13. Scoped to *one* non-cosine schedule, one size, one onset, one skill: write "replicates on a second schedule", never "schedule-general". The WSD−cosine difference (−0.221, Welch p = 0.016, Holm 0.032) is **read off the MDE (0.092), not off the p**. Never assert the two deficits are *equal* — presence is resolved, equality is not. WSD's higher absolute endpoints on both arms are descriptive only |
| 23 | `lr_shift` decomposes into a **level** component that does the reopening and a **shape** component that prevents the forgetting: level-only lifts +0.181 at +1.86 ppl, shape-only +0.004, the flat late-level control −0.009, and both together +0.196 at −0.33 ppl | C2 | **ESTABLISHED** — added 2026-08-13. Explains why only one arm reopened: it is the sole cell supplying both components. **Level-only is scored as forgetting, not as a second reopening** — it clears the skill gate but breaches the +1.0 ppl threshold, which §2's operational definition classifies as forgetting. The flat control C is load-bearing: without it, level-only confounds "higher LR" with "no decay". Holm p (A 0.092, both 0.075) are not the basis — the gate is a decision rule |
| 24 | The 60M syntax rate-collapse test **excludes a full collapse** of the high-acquisition mode (power 1.000) and **does not exclude a half collapse** (power 0.188, ~59 seeds/group), at 24 delayed vs 24 baseline seeds: 6/24 in both arms, Fisher p = 0.630, Mann–Whitney p = 0.626, both Holm 1.000, mean difference +0.0010 [−0.0064, +0.0087] | C1 × C3 | **BOUNDED-NOT-EXCLUDED** — added 2026-08-13. **Not a sixth route to the syntax null**: under the §10 cross-method rule an underpowered probe that agrees is not corroboration, and this one is underpowered against precisely the surviving alternative. It bounds claim 11b's scope; it does not promote it. Claim 11 stays at *five* probes |

---

## 4. Families and multiplicity

Confirmatory tests are Holm-corrected **within** the declared family and reported
uncorrected-plus-adjusted; exploratory tests are flagged and never carry a headline.

| family | members | policy | outcome |
|:--|:--|:--|:--|
| **F1 Detect** | paired deficit at 23M, 60M | Holm (m = 2) | 0.0107, 0.0087 — both survive |
| **F2 Reopen lift** | lr_shift lift over own null at 23M, 60M | Holm (m = 2) | 0.0065, 0.0054 — both survive |
| **F3 Reopen gate** | gate pass/fail per size | threshold rule fixed in advance; no p, no correction | both pass |
| **F4 Syntax** | the compound size-sweep gate | the gate *is* the decision rule; components are not independent tests | DISCONFIRM |
| **F5 Circuit** | one primary (ρ, exact p) | single test | p = 0.068 (n.s.) |
| **F6 SVA slopes** | paired attraction−complexity contrast, one per mixture | Holm (m = 2) | syntax survives; arithmetic does not |
| **F7 WSD schedule** | primary: WSD paired deficit ≠ 0; secondary: WSD deficit vs cosine deficit | Holm (m = 2), declared separately from F1–F3 | 0.477 n.s.; −0.221 raw 0.016 → Holm 0.032, survives. **Read off the MDE (0.092), not the p** |
| **F8 LR decomposition** | the three new 2×2 cells' paired lifts vs own null | Holm (m = 4, including lr_shift) | A 0.092, lr_shift 0.075, B 0.771, C 0.491 — none survives Holm; **the gate, not the p, is the decision rule** |
| **F9 Rate collapse** | Fisher exact at the pre-derived threshold; threshold-free Mann–Whitney | Holm (m = 2), pre-registered | 0.630, 0.626 → both 1.000. The 0.57 threshold was **excluded a priori** as unpowered by construction and never tested |
| **exploratory** | 11M H2/H3 · Week-8 seed-2/3 bootstrap inversion · trajectory (p = 0.053) · per-size cross-mixture contrasts · **§6's remaining tests: attraction magnitude at a single size (8.6e-05), the three mismatch-vs-chance tests, the complexity-slope null (0.43)** | flagged in-text, no headline | — |

**§6 scope, added 2026-08-11.** F6 is the *only* confirmatory family in §6; everything
else that section reports is exploratory **by declaration**, and §6's header now says so
rather than leaving it to be inferred from which numbers carry an adjustment. The audit
that prompted this: F6 and the exploratory row were correctly declared here, but the
manuscript's §2.7 named only "the agreement-slope family" and §6 itself carried no scope
line, so a reader could not tell which of that section's dozen *p* values were inside a
family and which were not. A correction policy that is declared in the backbone but not
legible in the section is not, from the reader's side, declared at all — the same
propagation failure as Appendix B's, in the reporting direction rather than the code one.

---

## 5. Measurement-consistency rules (whole-paper)

These exist because the numbers come from two eval harnesses and several run generations.

1. **One held-out eval set exists; anything else is contaminated.**
   `make_eval_set(ArithConfig(seed=1), 1000)` = `data/skills/arith/eval.jsonl` is the
   only problem set excluded from `train.bin`. A draw at any other seed overlaps the
   200 000 training problems by ~17–19 %. This is now enforced by a fail-loud guard in
   `week8_circuit_identity.py`; the same pattern remains unfixed in `week8_causal_trace.py`
   and `week8_pruning_feasibility.py`, neither of which supports a surviving claim.
   **Measurement floor on identical items is ~0.005** (mean |Δ| 0.0040, max 0.0080 across
   all 18 model-arm comparisons after the correction) — the two harnesses agree; the
   earlier ~0.03 figure was the contaminated draw, not harness disagreement. Accuracies
   may therefore be compared across the two tables down to ~0.01, but no finer.
2. **run_id is the identity of a model.** `arith_delay50_exp20_size23M_seed1` (endpoint
   0.506) and `…_seed1_recheck` (0.524, the canonical Detect null) are *different trained
   models*, not two readings of one. The Week-8 triad legitimately uses the former, since
   its C arm was forked from that checkpoint's step 2500. Do not substitute 0.524 into
   the circuit figure, or 0.506 into the Detect table.
3. **Every ± is an SD across seeds** unless explicitly labelled CI. The Week-8 bootstrap
   CIs are **eval-item variance only** (n = 1 model per point) and must be labelled as
   such at every appearance.
4. **State once in Methods:** arithmetic = teacher-forced exact match at step 5000 on 500
   held-out problems; syntax = continuation-scored licensing interaction; general
   capability = held-out val perplexity, always reported as the three-number
   decomposition.
5. **PAIRED-NULL DEPENDENCY — claim 2 and the C4 size null ship together or not at all.**
   The two size comparisons (C1 §2.1, C4 §2.4) are structurally identical — same n = 3 per
   group, same Welch test, same two sizes, both non-significant — and carry **opposite**
   evidential weight, for one reason only: C1's MDE (0.144) sits *below* the deficit it
   is asked about (0.21), so its null constrains; C4's MDE (0.809) *exceeds* the entire
   0.681 span its outcome occupies, so its null constrains nothing. **Neither may appear
   in the paper without an explicit cross-reference to the other.** A reader who meets
   only C1 over-trusts C4; one who meets only C4 under-trusts C1. This is a *reporting
   dependency between two claims*, not a stylistic preference, which is why it lives here
   and is flagged on both endpoints in §3 and §8 — a drafting note attached to one section
   cannot enforce a constraint whose failure mode is the *other* section shipping alone.
   Concretely: the Detect section states the pair and forward-references; the circuit
   section opens with the back-reference. Check both directions before submission.

---

## 6. Number → owner map

One artifact owns each number. If two artifacts disagree, the owner wins and the other is
regenerated — the paper never carries two versions of a figure.

| numbers | owner |
|:--|:--|
| Detect deficits, Reopen endpoints/gates/lifts, Holm families, MDEs, exact-permutation ρ, LOO, cross-harness agreement | `scripts/week9_stats_backbone.py` → `logs/week9_stats/backbone_stats.{json,md}` |
| 60M scale-check gate verdicts, matched-LR test | `scripts/arith_scale_gate.py` → `logs/week5-7_arith_scale/gate_result.json` |
| 23M H2/H3 discriminator (claim 8b) | `scripts/week9_h2h3_23M.py` → `logs/week9_h2h3_23M/h2h3_result.{json,md}` |
| Week-4 per-seed endpoints, deficit table, cram decomposition | `logs/week4_close/results.md`, `logs/week4_arith_pilot/seeds23_replication_results.md` |
| Syntax size-sweep gate | `scripts/szsweep_gate.py` → `logs/szsweep_eval/gate_result.json` |
| Circuit transfer, deficits, controls, Jaccard | `logs/week8_circuit_identity/{seed*_cleaneval.json, scale_vs_deficit_cleaneval.json}` — the un-suffixed files are the **superseded contaminated** run, kept only for the correction record (`logs/week9_stats/cleaneval_recheck.md`) |
| seed-2 vs seed-3 transfer CIs | `logs/week8_circuit_identity/transfer_bootstrap_23M_cleaneval.json` |
| Patching flatness / method agreement | `logs/week8_circuit_identity/method_agreement.json` |
| Pruning feasibility (NO-GO) | `logs/week8_pruning_feasibility/results.md` |
| SVA quartet, scale, mixture replication | `logs/sv_agreement_probe/*.json` + `replication_60M_results.md` |
| SVA slope estimator sensitivity | `scripts/agreement_slope_imbalance_check.py` → `slope_imbalance_check.md` |

---

## 7. Banned phrasings (retired wordings, with their replacements)

**RE-AUDIT THE MIDDLE COLUMN, NOT JUST THE LEFT ONE.** This table is a propagation
mechanism, and on 2026-08-03 it was caught propagating an error. The row
"at matched deficit, transfer level does not differ by size (p = 0.39)" entered as an
**approved replacement** on 2026-07-28 — and from there was copied, correctly and
obediently, into four downstream files. Every one of those copies was the table working
as designed.

That failure mode is the inverse of the ones this table was built for. Those were
*correct* language failing to reach every site, where the damage is bounded by how few
places adopted it and the fix is to propagate harder. This one is the distribution
channel itself carrying bad content — so the damage scales with how *well* the channel
works, and adoption is not evidence of correctness but the mechanism of the harm. An
error here is strictly worse than the same error left local, because it arrives
pre-blessed: downstream authors stop evaluating it, which is exactly what a
banned-phrasings table is *for*. Trust is the amplifier.

Practical consequence: a "write instead" cell is a **claim**, subject to every rule in
§5 and every label in §3 — it is not merely a safer way of saying something. Before
submission, audit each replacement as if it appeared in the abstract; a replacement that
asserts a null must carry its MDE, one that asserts an effect must carry its n and its
label. When a replacement is retired, keep it in the left column marked as a former
replacement (see the row below) rather than deleting it — the sites that adopted it in
good faith need to be findable.

| never write | write instead | why |
|:--|:--|:--|
| "~20× stronger than syntax" | absolute endpoints ± SD for each | divides by a ~0 denominator |
| "recovered N % of the gap" (as headline) | absolute treated endpoint ± SD **beside** the null ± SD | the denominator is seed-variable |
| "cross-method validated" (circuit) | "single-method, with three methods shown non-discriminating" | no second method discriminates on this contrast |
| "patching corroborates the reorganization" | — (retracted) | flat across all six triads, r = 0.045 |
| "scale drives reorganization" | "at matched deficit the size comparison is estimable but underpowered (p = 0.39, MDE 0.809 vs a 0.681 outcome span) — untested, not null" | on corrected numbers the deficit ranges overlap, so the question is no longer unidentifiable; it is simply unanswered at n = 3 per group |
| "at matched deficit, transfer level does not differ by size (p = 0.39)" | as above — the MDE must travel with it | *this was itself the 2026-07-28 replacement wording*; it asserts a bare null whose MDE exceeds the entire measured range, which §5 forbids |
| "deficit depth grades circuit identity" (as a finding) | "the direction persists but is not resolvable at n = 6 (ρ = −0.71, p = 0.068, LOO-fragile)" | contamination correction removed its significance |
| "the divergence did not replicate" / "is excluded" | "does not survive replication; not excluded (CI contains +0.133)" | CI still admits the original effect |
| "no critical period at 11M" | "excluded by the pre-registered floor: a capacity/budget boundary" | the anchor never plateaued |
| "size does not matter for the deficit" | "similar magnitude at both sizes (MDE 0.144)" | non-significance ≠ equivalence |
| "reset fails to reopen" | "no consistent direction at n = 3 (per-seed lifts disagree in sign)" | **replacement REVISED 2026-08-13**: the old wording "no detectable recovery at n = 1 (inside the null's cross-seed range)" is now itself stale — the arms are at n = 3 and fail on sign disagreement, not on insufficient n. Any site still carrying the n = 1 phrasing is out of date |
| "the deficit is the same under both schedules" | "abolished under WSD (−0.007 ± 0.013) against 0.214 ± 0.055 under cosine; the two deficits' *equality* is not resolved at this n" | presence is well-resolved, equality is not — MDE 0.092 |
| "the schedule mechanism is schedule-general" | "replicates on a second, deliberately dissimilar schedule" | one non-cosine schedule, one size, one onset, one skill |
| "level-only also reopens" | "level-only clears the skill gate but breaches the +1.0 ppl threshold, so it scores as forgetting under §2, not as reopening" | recovery at the cost of general capability is the pitfall the study was built to catch |
| "the rate test is a further route to the syntax null" | "excludes a full collapse (power 1.000); does not exclude a half collapse (power 0.188, ~59 seeds/group)" | an underpowered probe that agrees is not corroboration (§10) |

---

## 8. NOT-A-CLAIM (reported, supports nothing)

- 11M Detect paired deficit p = 0.0345 — pre-registered FLOOR exclusion (§2.1).
- **The converse manipulation (early window under the late LR segment), n = 1.**
  `arith_delay0_exp20_size23M_seed1_burst50lrlate`: arithmetic 0.778 → 0.142 at the
  window end, the direction the schedule account predicts, but val_ppl degrades with it
  (69.2 → 87.9). The early window *is* the from-scratch base-LM construction phase, so
  the manipulation damages the substrate and the arithmetic collapse cannot be separated
  from the general one. **The confound is structural, not statistical — more seeds
  replicate it rather than resolve it**, so this is not a candidate for the
  control-arm-seed-depth rule (§10). Report it (it is the manipulation every reader
  will ask for) and let the forward arm carry the evidence.
  Provenance: `logs/week4_arith_pilot/findings.md` "Converse run — RAN, but CONFOUNDED".
- 11M Reopen gate PASS — degenerate gate; the arm exceeds its own anchor (§2.2).
- d25 / d55 / d75 delay-dose points — n = 1 (23M seed 1), descriptive only, plotted
  without a fitted trend.
- Within-60M Spearman — mechanical at n = 3.
- Scale as an independent driver of circuit identity — now **estimable** at matched
  deficit (the deficit ranges overlap, so it is no longer confounded) but **underpowered**:
  p = 0.39 with MDE 0.809 against a 0.681 outcome span. Report as *untested* — neither
  "withdrawn" (the old confound wording) nor "a tested null" (the 2026-07-28 wording,
  which over-read estimability as power). **Paired-null dependency (§5.5): this entry and
  claim 2 must cross-reference each other; the contrast between their MDEs is what shows
  the paper reads non-significance off the MDE rather than off the p.**
  **⚠ How §5.5 behaves if §5 is cut for length (checked 2026-08-09,
  `logs/cut_dependencies.md`): the rule is SATISFIED, not violated.** §5.5 forbids either
  size null shipping without the other, and the underpowered half is this entry — §5's. A
  §5 deletion removes both halves at once, leaving claim 2 with its own MDE and no partner
  to owe. Recorded here because the natural reading of §5.5 on encountering a §5 cut is the
  opposite one ("we are breaking a rule"), and that misreading would either block a valid
  cut or trigger a pointless rewrite of §3.1. What §5.5 does still require in that case is
  that §1's MDE reporting rule lose its worked example, which pointed at §5.3.
- The 23M-only extrapolation onto the 60M points — no longer supports anything
  (residuals −0.30 / −0.15 / −0.20).
- Any number from `week8_causal_trace.py` or `week8_pruning_feasibility.py` — measured on
  contaminated draws and attached to no surviving claim.
- Circuit redundancy / sufficiency — untested, not supported.
- "±0.02 run noise" — never characterised; not a threshold anywhere.

---

## 9. What would move a label

Pre-set, so a future run cannot be read opportunistically.

- **60M arm-extension** completing its anneal past step 5000 → moves claim 10 from
  BOUNDED-NOT-EXCLUDED to a point estimate (either direction).
- **3+ further 11.5M SVA seeds** → moves claim 18 (arithmetic) from SUGGESTIVE toward
  ESTABLISHED if the robust p drops below 0.05 with the leverage rebalanced.
- **Joint head-group patching** with demonstrated dynamic range on the repair-vs-
  reorganized contrast → would let claims 13–15 drop the `(single-method)` annotation.
- **More triads (additional seeds at either size)** → the only thing that would move
  claim 15 (deficit depth grades identity) out of BOUNDED-NOT-EXCLUDED in either
  direction; at n = 6 with ρ = −0.71 the question is open, not answered. This is the one
  place where extra runs would still change a headline.
- **14 delayed 60M syntax seeds** (rate-collapse test at thr 0.5593 against the existing
  24-seed d0 baseline; 6 suffice at the permissive 0.55) → would move claim 11b's
  *differential-susceptibility* half from un-run to decided, in either direction. This is
  the cheapest outstanding test in the project and the only one that can promote 11b
  beyond ESTABLISHED (scoped). Assumes delay eliminates the high mode; if it only halves
  it, 59 are needed. The equivalent test at 23M needs 82 and is out of reach — run it at
  60M or not at all.
- Nothing else currently on the table changes a label; the remaining work is writing.

---

## 10. Standing rule — audit the ledger against the ORIGINAL GOALS

Sections 3–8 are *internal* consistency machinery: every claim labelled, labels matching
across abstract/section/discussion, no banned phrasing, every number owned. **None of it
can detect a missing row.** There is nothing inconsistent about a claim that was never
written down, so an omission passes every check in this document.

**Before submission, walk the master plan's goals and the pre-registered predictions one
at a time and ask "which ledger row answers this?"** — goals → ledger, not ledger →
goals, which §3 already covers. Record all three outcomes:

- **(a)** a row answers it;
- **(b)** no row does and one should exist → write it, with a label (this produced 11b);
- **(c)** the project answered a *different* question than the goal stated → say so
  explicitly instead of letting a near-miss read as a hit.

**(c) is the failure mode to watch.** The master plan's hit-criterion #1 was "a clean
dissociation: some skill/intervention combos recover, others don't", illustrated as
*"arithmetic is reopenable via embedding reset, syntax isn't"* — a **reopening**-side
dissociation. What the data support is a **detection**-side one: syntax never showed a
deficit, so no syntax reopening was attempted, and embedding reset turned out inert at
n = 1 even for arithmetic. Superficially the same finding; not the same claim. Writing the
plan's version would be unsupported.

Known open item from the 2026-08-03 pass: predictions #2 (reset most effective) and #3
(reset ⇒ "new circuit") are **not** ledger rows — #2 is covered obliquely by claim 9
(SUGGESTIVE, n = 1) and #3 by claims 13–15. Both were falsified or unaddressed rather
than confirmed; the methods/discussion must state that directly, since a reader holding
the pre-registration will look for them.

Companion: the §7 rule to re-audit *approved replacements*, and the CLAUDE.md standing
rule of the same name. All three exist because a trusted artifact distributes whatever it
contains — including what it omits.

---

## 11. Required standalone statements (pre-registration accountability)

Each pre-registered prediction gets **its own sentence, naming the prediction and its
outcome**, at the stated location. A reader holding the hypothesis document will search
for all three; addressing one obliquely through a claim row (as #2 currently is, via
claim 9) does not discharge it. These are required regardless of which anchor is chosen.

| pred | as pre-registered (CLAUDE.md) | outcome | required sentence must say | where | label it carries |
|--:|:--|:--|:--|:--|:--|
| **#1** | Syntax shows the *stronger* critical period — LD dependencies scaffolded by earlier structure, more fragile to a late start | **INVERTED** | Arithmetic carries the effect; LD shows none detectable under five probes. State the inversion explicitly — it is the origin of claim 11b, not an aside | Abstract + Discussion §1 | 11b, ESTABLISHED (scoped) |
| **#2** | Embedding/layer reset is the *most effective* intervention, with the biggest forgetting risk | **NOT BORNE OUT** | Reset showed no detectable recovery (0.586, inside the null's cross-seed range [0.524, 0.626]); `lr_shift` was the only arm to clear the gate. **Do not write "reset fails" as established** — reset is n = 1, so this is SUGGESTIVE; the honest form is "no detectable recovery at n = 1". The forgetting-risk half was also not observed as predicted | Discussion, intervention subsection | claim 9, SUGGESTIVE (n = 1) |
| **#3** | Circuit identity linked to #2 — *if reset wins, expect "new circuit" more often*, since reset destroys the original representation | **UNADDRESSED — antecedent never occurred** | Reset never produced recovery, so the conditional was never entered. C4 diagnoses `lr_shift`-recovered models instead, which the prediction did not cover. State that the prediction is untested rather than letting claims 13–15 read as its answer | Discussion, circuit subsection | none — explicitly untested |

**Why #3 needs care.** Claims 13–15 (repair↔reorganization, 60M reorganizes, deficit
grades identity) sit in the slot a reader expects prediction #3's answer to occupy, and
they are *about a different intervention*. Left unstated, they read as confirming or
disconfirming a prediction they cannot speak to. Name the mismatch.

This table is the output of the §10 goals→ledger audit; extend it if that audit finds
further case-(b) or case-(c) items.

---

## 12. Abstract anchors — DECIDED 2026-08-04

**Configuration: dissociation + mechanism, co-anchored. Claims 1, 11b, 4, 7.**

Two anchors answering *different* questions, which is why they co-anchor rather than
compete: **11b says what differs between skills**, **7 says why closure happens at all**.
Order in the abstract: 1 → 11b → 4 → 7 (phenomenon, then the contrast, then the
intervention, then the mechanism it implies).

| # | role in the abstract | label to honour |
|--:|:--|:--|
| 1 | the phenomenon exists and replicates (0.214 ± 0.055 at 23M, 0.229 ± 0.037 at 60M) | ESTABLISHED |
| 11b | the skill contrast — arithmetic pays, LD does not | ESTABLISHED **(scoped)** — scoping appears in-abstract, not deferred |
| 4 | it is reopenable (0.767 ± 0.014 / 0.796 ± 0.022 **beside** nulls 0.577 ± 0.051 / 0.600 ± 0.040) | ESTABLISHED |
| 7 | and reopening is schedule-driven, not entrenchment-driven (`cram+lr_shift ≈ lr_shift ≫ cram`) | ESTABLISHED |

**Required linking sentence.** Two anchors read as two papers without one. The link is
that both are *timing* results at different levels: 11b says timing damage is
skill-specific, 7 says the damage is carried by the optimizer's schedule rather than by
entrenched representations. Draft it explicitly; do not leave the reader to infer it.

**Two in-abstract obligations that fall out of the labels:**
- 11b's scoping (syntax's ~0.028 headroom vs arithmetic's 0.791) must appear in the
  abstract itself. An unscoped 11b in the abstract with the scope only in §2.3.1 violates
  the §1 rule that a label is identical across abstract, section and discussion.
- Claim 5 (scale-invariant recovered endpoint) is **not** an anchor but is the most
  surprising single number in C2; it belongs in the abstract's final clause or the
  Reopen section's lede, not dropped.

Prediction accountability (§11) is independent of this choice and still required.
