# Draft — §5 Circuit identity (draft 1, 2026-08-05)

Companion to `draft-sections-3-4.md`. Numbers from `logs/week8_circuit_identity/*_cleaneval.json`
(the corrected draw; the un-suffixed files are the superseded contaminated run) and
`logs/week9_stats/backbone_stats.md`. Budget ≈ 1.00 page.

> **RESOLVED 2026-08-05 — claim 13 reworded to the continuum form (backbone §2.4a).** The
> classifier's *repair* arm was a code-level default gated on `jaccard ≥ 0.4`, never
> pre-registered, contradicting its own docstring, and formally demoted to "corroborative
> only" on 2026-07-26 without the code being updated. This draft reports the continuous
> transfer axis and no categorical verdict of its own.

---

# §5 Did the intervention repair the original circuit, or build a new one?

Recovery tells us the skill came back; it does not tell us whether it came back *in the same
machinery*. For each seed and size we compare a triad that shares an initialization: **A**,
the early-exposure model; **B**, the delayed model; and **C**, the `lr_shift`-recovered model
forked from B. Each model's arithmetic circuit is localized by mean-ablating attention heads
one at a time and recording exact-match damage. The decisive test is reciprocal: ablating
**A's** top heads inside **C** measures how much of A's circuit is still load-bearing in the
recovered model, and ablating **C's own** top heads inside C controls for C simply being
robust to any ablation. We report the transfer ratio (the damage A's heads do in C as a
fraction of the damage they do in A) where 1.0 means C inherited A's circuitry intact and 0
means A's heads are causally inert in C. One of the two size comparisons below is the
underpowered counterpart to §3.1's well-powered one; §5.3 makes that pairing explicit.

All numbers are computed on the single held-out problem set. An earlier version of this
analysis drew its eval items per-run, which overlapped training by 17–19 % in four of the six
triads; all six were re-scored, the seed-1 triads reproduce exactly as a no-op control, and
the correction is reported in §2.4 because it changed a conclusion.

## 5.1 Only one of four probes discriminates on this contrast

Before reading anything off the ablation numbers, it matters which methods can tell the two
outcomes apart *at all*. Four were tried and three cannot:

- **Whole-layer mean-ablation saturates**, ablating any core layer drives accuracy to
  approximately zero in every model, so it separates nothing. Localization is therefore done
  at head granularity.
- **Linear CKA** of the residual stream sits at 0.66–0.83 across all six triads regardless of
  what the causal test says. Because the triads share an initialization, global geometry
  stays similar even where the wiring differs; CKA would have scored the most reorganized
  triad in the set at 0.658 and missed it entirely.
- **Single-head activation patching** is flat: its restoration ratio spans 0.591–0.792
  (SD 0.083) against the ablation transfer's 0.187–0.868 (SD 0.265), and the two are
  uncorrelated (*r* = 0.045, *p* = 0.93; recomputing on the corrected transfers gives
  *r* = −0.032, so the flatness is not an artifact of the contamination). The cause is a
  floor: single-head restoration is 0.05–0.15 in every model, so the ratio divides two small
  numbers and has no resolving power. An earlier draft cited patching as independent
  corroboration on the strength of one seed; running it on all six withdrew that.
- **Pruning-based localization** was pre-screened and rejected: mask overlap spans only 0.088
  across the triads, and its agreement with ablation *flips sign* across the sparsity sweep.
  A method whose sign is a hyperparameter choice cannot adjudicate (Appendix A).

We therefore report this cluster as **single-method by demonstration**, not by assumption.
Three non-discriminating probes is itself the evidence for that label: the alternative,
citing agreement from a method never shown to resolve the contrast, is how the retracted
patching claim happened.

## 5.2 Recovered circuits span a continuum, and it runs backwards at the shallow end

**Table 4.** Reciprocal ablation transfer, ordered by deficit depth. Transfer = damage from
A's heads in C ÷ their damage in A. Control = damage from C's own heads ÷ damage from A's
heads in C, so a high control confirms C has a working circuit of its own.

| size | seed | deficit | transfer | control | Jaccard(A, C) | pre-registered verdict |
|:--|--:|--:|--:|--:|--:|:--|
| 23M | 2 | 0.182 | 0.659 | 1.50 | 0.00 | mixed / partial |
| 60M | 2 | 0.202 | 0.404 | 2.43 | 0.00 | mixed / partial |
| 23M | 3 | 0.210 | 0.868 | 1.16 | 0.20 | mixed / partial |
| 60M | 1 | 0.224 | 0.445 | 2.18 | 0.09 | mixed / partial |
| 60M | 3 | 0.260 | 0.201 | 5.13 | 0.00 | new circuit |
| 23M | 1 | 0.292 | 0.187 | 5.36 | 0.00 | new circuit |

Recovered models do not behave alike. They span a continuum from near-complete transfer of
the original circuit to full reorganization. At one end (23M seed 3) the recovered model
inherits 87 % of the damage A's circuit takes, and its own top heads are barely more damaging
than A's: the original machinery is still carrying the skill. At the other (23M seed 1, 60M
seed 3) A's heads are close to inert in C while C's own heads are five times more damaging:
the skill returned in different machinery. **Every 60M triad sits at or below 0.445**, so at
the larger size reorganization is what we observe at every seed.

We report this axis continuously and assign no categorical verdict. Our implementation would
have labelled a triad *repaired* only on a conjunction of high transfer and high
top-head-set overlap; the overlap condition is never met here (the maximum is 0.20), so the
label was unreachable. It was an implementation default rather than a registered criterion,
and relabelling under a corrected rule would be choosing a decision rule after seeing the
number it decides (Appendix B). The mismatch is itself the methodological point: a recovered
model's head set can overlap its baseline's almost not at all while the baseline's heads
remain causally load-bearing: near-zero overlap at transfer 0.868. **Set overlap and causal
overlap are different measurements**, and only the second answers the question. That is the
case for reciprocal ablation over any similarity-of-components measure, and the same failure
that makes CKA useless here, from a second direction.

The ordering suggests that deficit depth grades this outcome (deeper deficits reorganize
more), but the evidence does not survive its own robustness check. Across the six triads
Spearman ρ = −0.714 with an exact permutation *p* = 0.0681, and leave-one-out is fragile:
four of the six single-point deletions push *p* above 0.05. The direction persists and is not
resolvable at *n* = 6. More triads at either size are the only thing that would settle it,
and this is the one place in the project where additional runs would still move a headline.

One pair of triads shows the relationship running backwards, and it is the best-measured
comparison in the table. At 23M, seeds 2 and 3 have nearly the same deficit (0.182 vs 0.210),
yet the *deeper* one shows *more* repair (0.868 vs 0.659), with non-overlapping bootstrap
intervals: [0.832, 0.903] against [0.611, 0.708]. Those intervals reflect eval-item variance
with one model per point (they say the two measurements differ, not that two populations do), but within the shallow-deficit regime the severity ordering inverts. Whatever grades
circuit identity is not deficit depth alone.

## 5.3 On size, this design is underpowered, unlike the one in §3.1

Transfer averages 0.571 ± 0.349 at 23M and 0.350 ± 0.131 at 60M (Welch *p* = 0.39). **That
non-significance means nothing here.** The minimum detectable effect at 80 % power is 0.809,
against an outcome span of only 0.681 across all six triads: no difference this design could
have produced would have been detected. The question is *untested*, not answered, and it is
no longer unidentifiable, since the corrected deficit ranges at the two sizes overlap, so
size is separable from deficit depth in a way it was not before the contamination fix.

This is worth setting beside §3.1's size comparison, which is structurally identical (same
*n* = 3 per group, same Welch test, same two sizes, also non-significant), and carries
opposite weight. There, the MDE (0.144) sits below the deficit being asked about (0.21), so
the null genuinely excludes a large size effect. Here it exceeds the entire measured range.
The two together are the clearest illustration of a rule this paper applies throughout:
non-significance is read off the minimum detectable effect, never off the *p*.

What the size axis does show is **variance**. The 60M transfers are tightly clustered
(SD 0.131) where the 23M ones are not (SD 0.349), a 2.7× ratio, mirroring §4's finding that
the intervention collapses cross-seed variance in the endpoint. We report this as suggestive:
it is a single comparison at *n* = 3 per group on an observational rather than manipulated
factor.

## 5.4 What these results do not speak to

Our third pre-registered prediction linked circuit identity to the reset intervention: if
reset were the winning intervention, we expected it to build a new circuit. Reset never
produced recovery (§4.4), so the antecedent never occurred and the prediction was never
entered. The models diagnosed here are `lr_shift`-recovered, which that prediction did not
cover; §7.2 states this rather than letting the section read as an answer to it.

Three further quantities appear in our released logs and support nothing. The within-60M
Spearman is mechanical at *n* = 3. An extrapolation from the 23M triads onto the 60M points
was built before the contamination correction and no longer fits (residuals −0.30, −0.15,
−0.20). Numbers from the causal-tracing and pruning scripts were computed on the contaminated
draws and were never re-run, since neither supports a surviving claim. Circuit redundancy and
sufficiency are likewise untested here: the retracted patching analysis was the only probe
that addressed them.

### §5 sentence → claim audit

| passage | claim | label | obligation honoured |
|:--|:--|:--|:--|
| triad design, transfer + control definitions | — | — | control stated before results, not after |
| contamination paragraph | — | — | §2.4 correction visible where its numbers are used |
| four probes, three non-discriminating | **13–15** | annotation | single-method *demonstrated*; retraction disclosed |
| Table 4, continuum, 87 % vs inert | **13** | ESTABLISHED (single-method) | continuum form per backbone §2.4a; no categorical verdict asserted |
| Jaccard 0.20 max; set vs causal overlap | — | — | classifier limitation disclosed in-text, not omitted |
| every 60M triad ≤ 0.445 | **14** | ESTABLISHED (single-method) | stated as observed at every seed |
| ρ = −0.714, exact p = 0.0681, LOO 4/6 | **15** | BOUNDED-NOT-EXCLUDED | banned "grades identity" wording avoided; direction + non-resolution |
| seeds 2/3 inversion, disjoint CIs | **16b** | ESTABLISHED (single-method) | CIs labelled eval-item variance, n = 1 model per point |
| Welch p = 0.39 + MDE 0.809 vs span 0.681 | — | NOT-A-CLAIM | MDE travels with the p; "untested, not null"; §5.5 pairing delivered |
| SD 0.349 vs 0.131, 2.7× | **16** | SUGGESTIVE | n = 3, observational factor, single comparison |
| prediction #3 never entered | — | — | §11 mismatch named, not implied |
| within-60M ρ, extrapolation, traced/pruned numbers, redundancy | — | NOT-A-CLAIM | all four placed, none left to be rediscovered |

---

## Resolved — claim 13 wording and the classifier's provenance

**MOVED TO SUPPLEMENTARY 2026-08-07 → `notes/draft-appendix.md` Appendix B.** The four-point
provenance record (never pre-registered; a code-level default; contradicts its own docstring
asymmetrically; formally demoted on 2026-07-26 without the code being updated) now lives
there, together with the reason relabelling was rejected. Appendices do not count against
the 8-page limit at this venue.

**What §5.2 keeps, and must keep** — because reviewers are not obliged to read an appendix:
the axis is reported continuously with no categorical verdict; the repair label was
unreachable in this dataset (max overlap 0.20); the rule was an implementation default, not
a registered criterion; and relabelling would be choosing a rule after seeing the number it
decides. The **set-overlap-versus-causal-overlap** point is *not* moved — near-zero overlap
at transfer 0.868 is the affirmative case for reciprocal ablation over any
component-similarity measure, and it is load-bearing for §5.1's CKA verdict too.

**Edits made:** backbone claim 13 row (continuum form, with a "do not write" note),
backbone §2.4a (provenance + the propagation lesson), and §5.2 above. **No abstract edit is
needed** — the abstract anchors are claims 1/11b/4/7 and it carries no circuit sentence.

## Length — 2026-08-07

Budget 1.00 page. Measured against the real template geometry (`scripts/page_estimate.py`):
**1.59 page before the appendix moves and 1.59 after.**

**The two moves bought nothing on the page, and that is the finding, not a failure of
execution.** Both are correct and both should stay: the pruning story and the classifier
provenance genuinely belong in supplementary material. But almost all of the *text* they
removed was already below the body cut-line — the provenance record lived in this file's
drafting apparatus, which the paper never contained. What was in the body was one bullet and
one paragraph, and the body must retain a self-sufficient version of each, because reviewers
are not obliged to read an appendix. Net movement: one line.

So §5's overrun is not made of removable narrative. It is Table 4 (7 rows), the four-probe
list, the MDE pairing owed to §3.1 under the paired-null dependency, and §5.2's analysis —
each of which carries a claim or an obligation. §5 is the largest single overrun in the
paper (+0.59) and the remaining options are content decisions, not edits. The nearest
candidate is §5.4's three inert quantities (~0.06 page), which exist precisely so that a
reader does not rediscover them and would have to move *with* their pointer.
