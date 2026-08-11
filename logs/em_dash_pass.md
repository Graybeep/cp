# Em-dash removal pass

Source: `scripts/drop_em_dashes.py`. Scope is the typeset ranges only
(eight sections, appendix, abstract); drafting apparatus in the same files is untouched.

**76 dashes rewritten across 64 lines. 27 flagged for review.**

A flagged site is one no rule could classify, so it took the fallback colon. Those are the
ones worth reading; the rest follow a conjunction, relative pronoun, participle or contrastive
tag, where the replacement is not a judgement call.

### REVIEW Abstract — `abstract-draft.md`:10
*UNCLASSIFIED -> colon, paired -> parentheses, relative clause*

```
- > Critical periods — windows after which a skill becomes hard to acquire regardless of
> further data — are usually studied in models whose pretraining history cannot be
> controlled. We train transformers of 11M–60M parameters from random initialization on a
> ~10M-word corpus, injecting synthetic arithmetic and long-distance *wh*-licensing on
> delay schedules that hold total skill exposure fixed and vary only *when* that exposure
> begins.
>
> Delaying arithmetic exposure to the midpoint of training produces a persistent
> endpoint deficit against seed-matched early-exposure anchors — 0.214 ± 0.055 at 23M and
> 0.229 ± 0.037 at 60M, in the same direction at every seed. Long-distance *wh*-licensing
> shows no detectable deficit under five distinct probes, inverting our pre-registered
> prediction that syntax would show the *stronger* critical period. We claim this
> dissociation in absolute terms and not as differential susceptibility, because the two
> skills are not equally observable: the syntax anchor sits 0.028 above its metric's
> chance floor against arithmetic's 0.791, so a proportionally matched deficit would have
> fallen below the design's resolution (MDE 0.048). Four of the five probes ask whether a
> critical period exists *given the acquisition that occurred*; only a representational
> probe — which carries independent dynamic range (selectivity 0.60, p < 1e-8) and is flat
> across timing (p = 0.48) — asks whether enough was acquired for one to have been
> detectable at all. That probe, not the count of nulls, is what the dissociation rests on.
>
> Where the deficit exists, it reopens. Shifting the learning-rate schedule at the delay
> onset restores endpoint accuracy to 0.767 ± 0.014 at 23M and 0.796 ± 0.022 at 60M,
> against untreated nulls of 0.577 ± 0.051 and 0.600 ± 0.040 — at 23M with no measurable
> cost to held-out perplexity, at 60M with a cost bounded above by 1.67 ppl that is
> plausibly an annealing-horizon artifact. Both recovered endpoints fall inside a
> pre-registered band, so recovery converges on a scale-invariant level rather than
> tracking capacity, and it collapses the cross-seed variance the untreated deficit shows.
> Concentrated re-exposure to the missed skill does not reproduce this: a data-only cram
> phase stays near the null, while cram combined with the schedule shift is
> indistinguishable from the shift alone. Closure at this onset is therefore largely a
> property of the optimizer's decayed learning-rate schedule rather than of entrenched
> representations.
>
> The two results are timing effects at different levels: *which* skills a late start
> damages is skill-specific, while the damage itself is carried by the optimizer's
> schedule rather than by representations that have hardened against revision.
+ > Critical periods (windows after which a skill becomes hard to acquire regardless of
> further data) are usually studied in models whose pretraining history cannot be
> controlled. We train transformers of 11M–60M parameters from random initialization on a
> ~10M-word corpus, injecting synthetic arithmetic and long-distance *wh*-licensing on
> delay schedules that hold total skill exposure fixed and vary only *when* that exposure
> begins.
>
> Delaying arithmetic exposure to the midpoint of training produces a persistent
> endpoint deficit against seed-matched early-exposure anchors: 0.214 ± 0.055 at 23M and
> 0.229 ± 0.037 at 60M, in the same direction at every seed. Long-distance *wh*-licensing
> shows no detectable deficit under five distinct probes, inverting our pre-registered
> prediction that syntax would show the *stronger* critical period. We claim this
> dissociation in absolute terms and not as differential susceptibility, because the two
> skills are not equally observable: the syntax anchor sits 0.028 above its metric's
> chance floor against arithmetic's 0.791, so a proportionally matched deficit would have
> fallen below the design's resolution (MDE 0.048). Four of the five probes ask whether a
> critical period exists *given the acquisition that occurred*; only a representational
> probe, which carries independent dynamic range (selectivity 0.60, p < 1e-8) and is flat
> across timing (p = 0.48): asks whether enough was acquired for one to have been
> detectable at all. That probe, not the count of nulls, is what the dissociation rests on.
>
> Where the deficit exists, it reopens. Shifting the learning-rate schedule at the delay
> onset restores endpoint accuracy to 0.767 ± 0.014 at 23M and 0.796 ± 0.022 at 60M,
> against untreated nulls of 0.577 ± 0.051 and 0.600 ± 0.040: at 23M with no measurable
> cost to held-out perplexity, at 60M with a cost bounded above by 1.67 ppl that is
> plausibly an annealing-horizon artifact. Both recovered endpoints fall inside a
> pre-registered band, so recovery converges on a scale-invariant level rather than
> tracking capacity, and it collapses the cross-seed variance the untreated deficit shows.
> Concentrated re-exposure to the missed skill does not reproduce this: a data-only cram
> phase stays near the null, while cram combined with the schedule shift is
> indistinguishable from the shift alone. Closure at this onset is therefore largely a
> property of the optimizer's decayed learning-rate schedule rather than of entrenched
> representations.
>
> The two results are timing effects at different levels: *which* skills a late start
> damages is skill-specific, while the damage itself is carried by the optimizer's
> schedule rather than by representations that have hardened against revision.
```

### Appendix — `draft-appendix.md`:24
*heading separator*

```
- # Appendix A — Pruning-based localization: the pre-screen that rejected it
+ # Appendix A. Pruning-based localization: the pre-screen that rejected it
```

### REVIEW Appendix — `draft-appendix.md`:32
*UNCLASSIFIED -> colon*

```
- **Method.** Learned binary gates over attention heads and MLPs, objective = arithmetic
answer-span cross-entropy + λ‖g‖₁, binarized at 0.5. The cross-model readout is
IoU(mask(A), mask(C)) — the overlap between the early model's sufficient subnetwork and the
recovered model's.
+ **Method.** Learned binary gates over attention heads and MLPs, objective = arithmetic
answer-span cross-entropy + λ‖g‖₁, binarized at 0.5. The cross-model readout is
IoU(mask(A), mask(C)): the overlap between the early model's sufficient subnetwork and the
recovered model's.
```

### REVIEW Appendix — `draft-appendix.md`:41
*UNCLASSIFIED -> colon*

```
- | 1. reproducibility | IoU(A, A′) ≥ 0.70 on two independent fits | **PASS** — 1.000, the two fits give the identical mask |
+ | 1. reproducibility | IoU(A, A′) ≥ 0.70 on two independent fits | **PASS**: 1.000, the two fits give the identical mask |
```

### REVIEW Appendix — `draft-appendix.md`:42
*UNCLASSIFIED -> colon*

```
- | 2. dynamic range | spread of IoU(A, C) ≥ 0.20 across seeds | **FAIL** — 0.088, all three inside one 0.10-wide band |
+ | 2. dynamic range | spread of IoU(A, C) ≥ 0.20 across seeds | **FAIL**: 0.088, all three inside one 0.10-wide band |
```

### REVIEW Appendix — `draft-appendix.md`:43
*UNCLASSIFIED -> colon*

```
- | 3. ordering | seed 1 (most reorganized by ablation) lowest | **FAIL** — seed 3 is lowest at 0.425; seed 1 is middle |
+ | 3. ordering | seed 1 (most reorganized by ablation) lowest | **FAIL**: seed 3 is lowest at 0.425; seed 1 is middle |
```

### REVIEW Appendix — `draft-appendix.md`:44
*UNCLASSIFIED -> colon*

```
- | 4. functional validity | mask retains ≥ 0.80 of accuracy | **PASS** — 0.80–0.94 on ~30/54 components; complement → 0.000 |
+ | 4. functional validity | mask retains ≥ 0.80 of accuracy | **PASS**: 0.80–0.94 on ~30/54 components; complement → 0.000 |
```

### REVIEW Appendix — `draft-appendix.md`:46
*UNCLASSIFIED -> colon*

```
- The metric is real: perfectly reproducible, and the mask genuinely is a sufficient
arithmetic subnetwork. It simply does not track circuit identity — A and C overlap about
the same amount whether or not C rebuilt.
+ The metric is real: perfectly reproducible, and the mask genuinely is a sufficient
arithmetic subnetwork. It simply does not track circuit identity: A and C overlap about
the same amount whether or not C rebuilt.
```

### REVIEW Appendix — `draft-appendix.md`:64
*UNCLASSIFIED -> colon*

```
- And the λ that agrees with the primary readout (*r* = +0.872) is exactly the λ at which the
mask stops being a valid circuit — retention falls to 0.67 / 0.70, below the pre-registered
0.80 bar. At that sparsity, overlap differences plausibly track how much accuracy each mask
sacrificed rather than which components each model uses.
+ And the λ that agrees with the primary readout (*r* = +0.872) is exactly the λ at which the
mask stops being a valid circuit: retention falls to 0.67 / 0.70, below the pre-registered
0.80 bar. At that sparsity, overlap differences plausibly track how much accuracy each mask
sacrificed rather than which components each model uses.
```

### Appendix — `draft-appendix.md`:69
*participial phrase*

```
- **The near-miss is the point.** Run at λ = 0.06 alone, this pilot would have reported "seed 1
lowest overlap, *r* = +0.87 with ablation — pruning independently confirms reorganization."
That is the same shape as the single-head patching result we retracted (§5.1), and wrong for
the same reason: a secondary method cited as corroboration without being shown to
discriminate across the full contrast set. It is why §5's claims are annotated single-method
by demonstration rather than supported by a count of agreeing probes.
+ **The near-miss is the point.** Run at λ = 0.06 alone, this pilot would have reported "seed 1
lowest overlap, *r* = +0.87 with ablation, pruning independently confirms reorganization."
That is the same shape as the single-head patching result we retracted (§5.1), and wrong for
the same reason: a secondary method cited as corroboration without being shown to
discriminate across the full contrast set. It is why §5's claims are annotated single-method
by demonstration rather than supported by a count of agreeing probes.
```

### Appendix — `draft-appendix.md`:78
*heading separator*

```
- # Appendix B — Provenance of the circuit-identity classifier's thresholds
+ # Appendix B: Provenance of the circuit-identity classifier's thresholds
```

### Appendix — `draft-appendix.md`:80
*paired -> parentheses*

```
- §5.2 reports the transfer axis continuously and assigns no categorical verdict. This
appendix records why, since the alternative — relabelling under a corrected rule — was
available and was rejected.
+ §5.2 reports the transfer axis continuously and assigns no categorical verdict. This
appendix records why, since the alternative (relabelling under a corrected rule) was
available and was rejected.
```

### Appendix — `draft-appendix.md`:107
*relative clause*

```
- **Why we did not simply relabel.** Dropping the overlap conjunct would move the
highest-transfer triad (23M seed 3, transfer 0.868) into a *repair* category — which is
choosing a decision rule after seeing the number it decides. Reporting the continuous axis
assumes neither rule, and the verdict column stays in the body's Table 4 so that the absence
of a repair row is visible rather than quietly dropped. The measurements are untouched
throughout; only the labelling was ever in question.
+ **Why we did not simply relabel.** Dropping the overlap conjunct would move the
highest-transfer triad (23M seed 3, transfer 0.868) into a *repair* category, which is
choosing a decision rule after seeing the number it decides. Reporting the continuous axis
assumes neither rule, and the verdict column stays in the body's Table 4 so that the absence
of a repair row is visible rather than quietly dropped. The measurements are untouched
throughout; only the labelling was ever in question.
```

### Appendix — `draft-appendix.md`:114
*coordinating conjunction*

```
- **The generalizable lesson** is in the fourth point: a decision recorded in a
pre-registration does not propagate itself into code. The demotion of overlap to
corroborative was written down, correctly, ten days before this analysis was read — and the
classifier went on applying the superseded rule because nobody grepped the implementation in
the same sitting.
+ **The generalizable lesson** is in the fourth point: a decision recorded in a
pre-registration does not propagate itself into code. The demotion of overlap to
corroborative was written down, correctly, ten days before this analysis was read, and the
classifier went on applying the superseded rule because nobody grepped the implementation in
the same sitting.
```

### Appendix — `draft-appendix.md`:122
*heading separator*

```
- # Appendix C — Material moved from the body in the condensed version
+ # Appendix C: Material moved from the body in the condensed version
```

### REVIEW Appendix — `draft-appendix.md`:124
*UNCLASSIFIED -> colon*

```
- Appendices A and B are moves both versions make. **This appendix is different: it exists
because the condensed build moves material out of §2, §6 and §8 that the full version
carries inline.** Nothing here is a new result, and nothing here is load-bearing for a
claim — each item is evidence for a decision that is stated and justified in the body,
with the body keeping a self-sufficient version and a pointer.
+ Appendices A and B are moves both versions make. **This appendix is different: it exists
because the condensed build moves material out of §2, §6 and §8 that the full version
carries inline.** Nothing here is a new result, and nothing here is load-bearing for a
claim: each item is evidence for a decision that is stated and justified in the body,
with the body keeping a self-sufficient version and a pointer.
```

### REVIEW Appendix — `draft-appendix.md`:192
*UNCLASSIFIED -> colon*

```
- §6 reports that attraction may also decline *within* a single run and explicitly does not
claim it. The supporting numbers: between steps 2500 and 5000 the isolated attraction cost
falls +0.392 → +0.303 (paired, *n* = 6 at 23.2M, 5/6 seeds), marginal at *p* = 0.053, with
**53 % power against this effect size and an MDE of 0.125** — roughly ten seeds would
decide it. The mismatch cell itself improves significantly over the same interval
(+0.070, *p* = 0.041), and that is the non-marginal part of the result.
+ §6 reports that attraction may also decline *within* a single run and explicitly does not
claim it. The supporting numbers: between steps 2500 and 5000 the isolated attraction cost
falls +0.392 → +0.303 (paired, *n* = 6 at 23.2M, 5/6 seeds), marginal at *p* = 0.053, with
**53 % power against this effect size and an MDE of 0.125**: roughly ten seeds would
decide it. The mismatch cell itself improves significantly over the same interval
(+0.070, *p* = 0.041), and that is the non-marginal part of the result.
```

### Appendix — `draft-appendix.md`:214
*paired -> parentheses*

```
- If delay merely *halves* the mode rather than eliminating it, the requirement rises to 59
per group at 60M and 203 at 23M. The direction that matters is that feasibility **improves
with scale** — the high mode is roughly four times more common at 60M — which is why the
test is named at 60M and not at the size where the delayed contrast was actually run.
+ If delay merely *halves* the mode rather than eliminating it, the requirement rises to 59
per group at 60M and 203 at 23M. The direction that matters is that feasibility **improves
with scale** (the high mode is roughly four times more common at 60M) which is why the
test is named at 60M and not at the size where the delayed contrast was actually run.
```

### Related Work — `draft-related-work.md`:29
*coordinating conjunction*

```
- **Plasticity loss and warm starting.** A separate literature finds that networks become
harder to train as training proceeds: warm-started models generalize worse than freshly
initialized ones given the same final data [Ash & Adams, NeurIPS 2020], and continual-learning
agents progressively lose the ability to fit new tasks, recoverable by re-injecting
plasticity through resets or noise [Dohare et al., Nature 2024; Lyle et al.; Berariu et al.].
**This is the closest prior explanation of our reopening result and the one we most risk
rediscovering**, so we state the difference precisely: those settings involve a distribution
shift, a restart, or a sequence of distinct tasks, and typically recover plasticity by
perturbing *weights*. Our models train once, without interruption, on a stationary mixture
whose only manipulated variable is a skill's onset step — and the intervention that recovers
the skill changes no weights at all, only the learning-rate schedule, while resetting weights
(the standard plasticity remedy) produced no detectable recovery in our setting at *n* = 1.
The mechanisms may still be the same one seen from two directions; what we can say is that
the recovery here is obtainable without weight perturbation. (That reset arm is *n* = 1;
§7.2 states what it does and does not license.)
+ **Plasticity loss and warm starting.** A separate literature finds that networks become
harder to train as training proceeds: warm-started models generalize worse than freshly
initialized ones given the same final data [Ash & Adams, NeurIPS 2020], and continual-learning
agents progressively lose the ability to fit new tasks, recoverable by re-injecting
plasticity through resets or noise [Dohare et al., Nature 2024; Lyle et al.; Berariu et al.].
**This is the closest prior explanation of our reopening result and the one we most risk
rediscovering**, so we state the difference precisely: those settings involve a distribution
shift, a restart, or a sequence of distinct tasks, and typically recover plasticity by
perturbing *weights*. Our models train once, without interruption, on a stationary mixture
whose only manipulated variable is a skill's onset step, and the intervention that recovers
the skill changes no weights at all, only the learning-rate schedule, while resetting weights
(the standard plasticity remedy) produced no detectable recovery in our setting at *n* = 1.
The mechanisms may still be the same one seen from two directions; what we can say is that
the recovery here is obtainable without weight perturbation. (That reset arm is *n* = 1;
§7.2 states what it does and does not license.)
```

### §5 Circuit identity — `draft-section-5.md`:17
*paired -> parentheses*

```
- Recovery tells us the skill came back; it does not tell us whether it came back *in the same
machinery*. For each seed and size we compare a triad that shares an initialization: **A**,
the early-exposure model; **B**, the delayed model; and **C**, the `lr_shift`-recovered model
forked from B. Each model's arithmetic circuit is localized by mean-ablating attention heads
one at a time and recording exact-match damage. The decisive test is reciprocal: ablating
**A's** top heads inside **C** measures how much of A's circuit is still load-bearing in the
recovered model, and ablating **C's own** top heads inside C controls for C simply being
robust to any ablation. We report the transfer ratio — the damage A's heads do in C as a
fraction of the damage they do in A — where 1.0 means C inherited A's circuitry intact and 0
means A's heads are causally inert in C. One of the two size comparisons below is the
underpowered counterpart to §3.1's well-powered one; §5.3 makes that pairing explicit.
+ Recovery tells us the skill came back; it does not tell us whether it came back *in the same
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
```

### §5 Circuit identity — `draft-section-5.md`:39
*participial phrase*

```
- - **Whole-layer mean-ablation saturates** — ablating any core layer drives accuracy to
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
+ - **Whole-layer mean-ablation saturates**, ablating any core layer drives accuracy to
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
```

### REVIEW §5 Circuit identity — `draft-section-5.md`:57
*UNCLASSIFIED -> colon*

```
- We therefore report this cluster as **single-method by demonstration**, not by assumption.
Three non-discriminating probes is itself the evidence for that label — the alternative,
citing agreement from a method never shown to resolve the contrast, is how the retracted
patching claim happened.
+ We therefore report this cluster as **single-method by demonstration**, not by assumption.
Three non-discriminating probes is itself the evidence for that label: the alternative,
citing agreement from a method never shown to resolve the contrast, is how the retracted
patching claim happened.
```

### REVIEW §5 Circuit identity — `draft-section-5.md`:77
*UNCLASSIFIED -> colon*

```
- Recovered models do not behave alike. They span a continuum from near-complete transfer of
the original circuit to full reorganization. At one end (23M seed 3) the recovered model
inherits 87 % of the damage A's circuit takes, and its own top heads are barely more damaging
than A's — the original machinery is still carrying the skill. At the other (23M seed 1, 60M
seed 3) A's heads are close to inert in C while C's own heads are five times more damaging:
the skill returned in different machinery. **Every 60M triad sits at or below 0.445**, so at
the larger size reorganization is what we observe at every seed.
+ Recovered models do not behave alike. They span a continuum from near-complete transfer of
the original circuit to full reorganization. At one end (23M seed 3) the recovered model
inherits 87 % of the damage A's circuit takes, and its own top heads are barely more damaging
than A's: the original machinery is still carrying the skill. At the other (23M seed 1, 60M
seed 3) A's heads are close to inert in C while C's own heads are five times more damaging:
the skill returned in different machinery. **Every 60M triad sits at or below 0.445**, so at
the larger size reorganization is what we observe at every seed.
```

### REVIEW §5 Circuit identity — `draft-section-5.md`:85
*UNCLASSIFIED -> colon, paired -> parentheses*

```
- We report this axis continuously and assign no categorical verdict. Our implementation would
have labelled a triad *repaired* only on a conjunction of high transfer and high
top-head-set overlap; the overlap condition is never met here — the maximum is 0.20 — so the
label was unreachable. It was an implementation default rather than a registered criterion,
and relabelling under a corrected rule would be choosing a decision rule after seeing the
number it decides (Appendix B). The mismatch is itself the methodological point: a recovered
model's head set can overlap its baseline's almost not at all while the baseline's heads
remain causally load-bearing — near-zero overlap at transfer 0.868. **Set overlap and causal
overlap are different measurements**, and only the second answers the question. That is the
case for reciprocal ablation over any similarity-of-components measure, and the same failure
that makes CKA useless here, from a second direction.
+ We report this axis continuously and assign no categorical verdict. Our implementation would
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
```

### §5 Circuit identity — `draft-section-5.md`:97
*paired -> parentheses*

```
- The ordering suggests that deficit depth grades this outcome — deeper deficits reorganize
more — but the evidence does not survive its own robustness check. Across the six triads
Spearman ρ = −0.714 with an exact permutation *p* = 0.0681, and leave-one-out is fragile:
four of the six single-point deletions push *p* above 0.05. The direction persists and is not
resolvable at *n* = 6. More triads at either size are the only thing that would settle it,
and this is the one place in the project where additional runs would still move a headline.
+ The ordering suggests that deficit depth grades this outcome (deeper deficits reorganize
more), but the evidence does not survive its own robustness check. Across the six triads
Spearman ρ = −0.714 with an exact permutation *p* = 0.0681, and leave-one-out is fragile:
four of the six single-point deletions push *p* above 0.05. The direction persists and is not
resolvable at *n* = 6. More triads at either size are the only thing that would settle it,
and this is the one place in the project where additional runs would still move a headline.
```

### §5 Circuit identity — `draft-section-5.md`:104
*paired -> parentheses*

```
- One pair of triads shows the relationship running backwards, and it is the best-measured
comparison in the table. At 23M, seeds 2 and 3 have nearly the same deficit (0.182 vs 0.210),
yet the *deeper* one shows *more* repair (0.868 vs 0.659), with non-overlapping bootstrap
intervals: [0.832, 0.903] against [0.611, 0.708]. Those intervals reflect eval-item variance
with one model per point — they say the two measurements differ, not that two populations do
— but within the shallow-deficit regime the severity ordering inverts. Whatever grades
circuit identity is not deficit depth alone.
+ One pair of triads shows the relationship running backwards, and it is the best-measured
comparison in the table. At 23M, seeds 2 and 3 have nearly the same deficit (0.182 vs 0.210),
yet the *deeper* one shows *more* repair (0.868 vs 0.659), with non-overlapping bootstrap
intervals: [0.832, 0.903] against [0.611, 0.708]. Those intervals reflect eval-item variance
with one model per point (they say the two measurements differ, not that two populations do), but within the shallow-deficit regime the severity ordering inverts. Whatever grades
circuit identity is not deficit depth alone.
```

### §5 Circuit identity — `draft-section-5.md`:112
*heading separator*

```
- ## 5.3 On size, this design is underpowered — unlike the one in §3.1
+ ## 5.3 On size, this design is underpowered, unlike the one in §3.1
```

### §5 Circuit identity — `draft-section-5.md`:114
*coordinating conjunction*

```
- Transfer averages 0.571 ± 0.349 at 23M and 0.350 ± 0.131 at 60M (Welch *p* = 0.39). **That
non-significance means nothing here.** The minimum detectable effect at 80 % power is 0.809,
against an outcome span of only 0.681 across all six triads: no difference this design could
have produced would have been detected. The question is *untested*, not answered — and it is
no longer unidentifiable, since the corrected deficit ranges at the two sizes overlap, so
size is separable from deficit depth in a way it was not before the contamination fix.
+ Transfer averages 0.571 ± 0.349 at 23M and 0.350 ± 0.131 at 60M (Welch *p* = 0.39). **That
non-significance means nothing here.** The minimum detectable effect at 80 % power is 0.809,
against an outcome span of only 0.681 across all six triads: no difference this design could
have produced would have been detected. The question is *untested*, not answered, and it is
no longer unidentifiable, since the corrected deficit ranges at the two sizes overlap, so
size is separable from deficit depth in a way it was not before the contamination fix.
```

### §5 Circuit identity — `draft-section-5.md`:121
*paired -> parentheses*

```
- This is worth setting beside §3.1's size comparison, which is structurally identical — same
*n* = 3 per group, same Welch test, same two sizes, also non-significant — and carries
opposite weight. There, the MDE (0.144) sits below the deficit being asked about (0.21), so
the null genuinely excludes a large size effect. Here it exceeds the entire measured range.
The two together are the clearest illustration of a rule this paper applies throughout:
non-significance is read off the minimum detectable effect, never off the *p*.
+ This is worth setting beside §3.1's size comparison, which is structurally identical (same
*n* = 3 per group, same Welch test, same two sizes, also non-significant), and carries
opposite weight. There, the MDE (0.144) sits below the deficit being asked about (0.21), so
the null genuinely excludes a large size effect. Here it exceeds the entire measured range.
The two together are the clearest illustration of a rule this paper applies throughout:
non-significance is read off the minimum detectable effect, never off the *p*.
```

### §6 Agreement — `draft-section-6.md`:18
*paired -> parentheses*

```
- The models trained for this study support one question the critical-period design did not
set out to ask. Subject–verb agreement is acquired almost perfectly here — 0.98 on
controlled items — yet fails when a number-mismatching noun intervenes. A fully-crossed item
design, in which the verb-form frequency prior cancels *exactly*,
separates two components of that failure: **attraction**, the cost of an intervening noun
whose number mismatches the head, and **complexity**, the cost of any intervening material
at all.
+ The models trained for this study support one question the critical-period design did not
set out to ask. Subject–verb agreement is acquired almost perfectly here (0.98 on
controlled items), yet fails when a number-mismatching noun intervenes. A fully-crossed item
design, in which the verb-form frequency prior cancels *exactly*,
separates two components of that failure: **attraction**, the cost of an intervening noun
whose number mismatches the head, and **complexity**, the cost of any intervening material
at all.
```

### REVIEW §6 Agreement — `draft-section-6.md`:26
*UNCLASSIFIED -> colon*

```
- **Attraction is real, large, and the dominant component.** At 23.2M it costs
+0.303 ± 0.064 (*p* = 8.6e-05) against complexity's +0.143 ± 0.035, in both number
directions and both frame types. A unigram-frequency control on the same items returns
exactly 0.000 by construction, so the effect cannot be a verb-form prior. Attraction exceeds
complexity at every size and is positive in **81 of 81 models** across two independent
training mixtures — 72 syntax-enriched (3 sizes × 24 seeds) and 9 arithmetic-enriched.
+ **Attraction is real, large, and the dominant component.** At 23.2M it costs
+0.303 ± 0.064 (*p* = 8.6e-05) against complexity's +0.143 ± 0.035, in both number
directions and both frame types. A unigram-frequency control on the same items returns
exactly 0.000 by construction, so the effect cannot be a verb-form prior. Attraction exceeds
complexity at every size and is positive in **81 of 81 models** across two independent
training mixtures: 72 syntax-enriched (3 sizes × 24 seeds) and 9 arithmetic-enriched.
```

### REVIEW §6 Agreement — `draft-section-6.md`:38
*UNCLASSIFIED -> colon, paired -> parentheses*

```
- **Attraction declines with scale; whether it dissociates from complexity depends on the
mixture.** In the syntax-enriched mixture attraction falls **−0.316 per decade of
parameters** (*r* = −0.803, *p* = 2.3e-17, *n* = 72), from +0.430 to +0.201, while complexity
is **flat** (+0.016/decade, *p* = 0.43). The second mixture reproduces the attraction half
in direction and magnitude, **−0.207/decade**, stable at −0.21 to −0.24 under every
estimator tried but at **estimator-dependent significance — *p* = 0.033 under OLS, 0.073
under a heteroscedasticity-robust SE** — on an unbalanced (3, 3, 8) seed design whose
leverage sits on the under-sampled small end. It does **not** reproduce the dissociation:
complexity falls there too, and the paired within-model contrast is flat (−0.095,
*p* = 0.36) against −0.333 (*p* = 3.8e-11) in the syntax mixture, the one part of the
comparison that is not estimator-fragile. A cross-mixture divergence at 60.3M **does not
survive replication** at *n* = 8 (+0.065, *p* = 0.091, halved from +0.133) and **is not
excluded** — the 95 % CI still contains the original effect. An English-exposure explanation
for it is excluded separately: exposure is constant across sizes by design and can only
produce a uniform offset.
+ **Attraction declines with scale; whether it dissociates from complexity depends on the
mixture.** In the syntax-enriched mixture attraction falls **−0.316 per decade of
parameters** (*r* = −0.803, *p* = 2.3e-17, *n* = 72), from +0.430 to +0.201, while complexity
is **flat** (+0.016/decade, *p* = 0.43). The second mixture reproduces the attraction half
in direction and magnitude, **−0.207/decade**, stable at −0.21 to −0.24 under every
estimator tried but at **estimator-dependent significance (*p* = 0.033 under OLS, 0.073
under a heteroscedasticity-robust SE**) on an unbalanced (3, 3, 8) seed design whose
leverage sits on the under-sampled small end. It does **not** reproduce the dissociation:
complexity falls there too, and the paired within-model contrast is flat (−0.095,
*p* = 0.36) against −0.333 (*p* = 3.8e-11) in the syntax mixture, the one part of the
comparison that is not estimator-fragile. A cross-mixture divergence at 60.3M **does not
survive replication** at *n* = 8 (+0.065, *p* = 0.091, halved from +0.133) and **is not
excluded**: the 95 % CI still contains the original effect. An English-exposure explanation
for it is excluded separately: exposure is constant across sizes by design and can only
produce a uniform offset.
```

### REVIEW §6 Agreement — `draft-section-6.md`:54
*UNCLASSIFIED -> colon*

```
- **Within a single run, attraction may also decline over training, but we do not claim it.**
Between steps 2500 and 5000 the isolated attraction cost falls +0.392 → +0.303 (paired,
*n* = 6 at 23.2M, 5/6 seeds) — marginal at *p* = 0.053, with 53 % power against this effect
size and an MDE of 0.125; roughly ten seeds would decide it. The mismatch cell itself
improves significantly over the same interval (+0.070, *p* = 0.041). We report the direction
and the power, and treat the trajectory as suggestive only.
+ **Within a single run, attraction may also decline over training, but we do not claim it.**
Between steps 2500 and 5000 the isolated attraction cost falls +0.392 → +0.303 (paired,
*n* = 6 at 23.2M, 5/6 seeds): marginal at *p* = 0.053, with 53 % power against this effect
size and an MDE of 0.125; roughly ten seeds would decide it. The mismatch cell itself
improves significantly over the same interval (+0.070, *p* = 0.041). We report the direction
and the power, and treat the trajectory as suggestive only.
```

### REVIEW §6 Agreement — `draft-section-6.md`:61
*UNCLASSIFIED -> colon*

```
- **The standard metric gets this wrong at every scale, in three different ways.** Uncontrolled
BLiMP-style items report below-chance behaviour at all three sizes, which is true only at the
smallest; the frequency-controlled metric locates it correctly; and only the crossed design
separates attraction from complexity. Three metrics, three conclusions, on the same models.
A related trap is worth naming: pooling all 72 models into one attraction estimate averages
across a size-dependent effect and yields a number that describes no size — an
averaging-across-a-dissociation error, not a confound.
+ **The standard metric gets this wrong at every scale, in three different ways.** Uncontrolled
BLiMP-style items report below-chance behaviour at all three sizes, which is true only at the
smallest; the frequency-controlled metric locates it correctly; and only the crossed design
separates attraction from complexity. Three metrics, three conclusions, on the same models.
A related trap is worth naming: pooling all 72 models into one attraction estimate averages
across a size-dependent effect and yields a number that describes no size: an
averaging-across-a-dissociation error, not a confound.
```

### §1 Introduction — `draft-sections-1-2.md`:20
*paired -> parentheses*

```
- We take the other route. Every model here is trained from random initialization on a fixed
~10M-word corpus, with a target skill injected into the training stream on a schedule we
control. Conditions differ in *when* the skill arrives and in nothing else — total skill
tokens are held constant — so a difference in final performance is a timing effect, not a
dose effect. This costs us scale, the largest model being 60.3M parameters, and buys the one
thing a pretrained checkpoint cannot provide: knowing when the model first saw arithmetic.
+ We take the other route. Every model here is trained from random initialization on a fixed
~10M-word corpus, with a target skill injected into the training stream on a schedule we
control. Conditions differ in *when* the skill arrives and in nothing else (total skill
tokens are held constant), so a difference in final performance is a timing effect, not a
dose effect. This costs us scale, the largest model being 60.3M parameters, and buys the one
thing a pretrained checkpoint cannot provide: knowing when the model first saw arithmetic.
```

### REVIEW §1 Introduction — `draft-sections-1-2.md`:27
*UNCLASSIFIED -> colon, participial phrase*

```
- Within that design we find, first, that **delaying arithmetic exposure to the midpoint of
training leaves a deficit that the remaining half of training does not close**, at both
confirmatory sizes and at every seed. Second, that **long-distance *wh*-licensing shows no
such deficit** under five distinct probes — inverting our pre-registered prediction that
syntax, the more structurally scaffolded skill, would be the more fragile one. We state that
dissociation in absolute terms rather than as differential susceptibility; §3.3 gives the
measurement reason. Third, that **the deficit is reopenable**: restoring the learning-rate
schedule at the point of delayed exposure recovers the endpoint to within 90 % of an
early-exposure anchor at both sizes, and to nearly the same absolute level at each. Fourth,
that **concentrated re-exposure does not reproduce this** — extra skill density on top of a
restored schedule adds nothing, while a restored schedule without it recovers the skill.
What closes the window at this onset is therefore largely the optimizer's decayed
learning-rate schedule rather than representations hardened against revision.
+ Within that design we find, first, that **delaying arithmetic exposure to the midpoint of
training leaves a deficit that the remaining half of training does not close**, at both
confirmatory sizes and at every seed. Second, that **long-distance *wh*-licensing shows no
such deficit** under five distinct probes, inverting our pre-registered prediction that
syntax, the more structurally scaffolded skill, would be the more fragile one. We state that
dissociation in absolute terms rather than as differential susceptibility; §3.3 gives the
measurement reason. Third, that **the deficit is reopenable**: restoring the learning-rate
schedule at the point of delayed exposure recovers the endpoint to within 90 % of an
early-exposure anchor at both sizes, and to nearly the same absolute level at each. Fourth,
that **concentrated re-exposure does not reproduce this**: extra skill density on top of a
restored schedule adds nothing, while a restored schedule without it recovers the skill.
What closes the window at this onset is therefore largely the optimizer's decayed
learning-rate schedule rather than representations hardened against revision.
```

### REVIEW §1 Introduction — `draft-sections-1-2.md`:41
*UNCLASSIFIED -> colon*

```
- Two things follow for how we report. A delayed model's endpoint is strongly seed-dependent
while its early-exposure anchor is not, so we report absolute endpoints beside their nulls
rather than recovery as a percentage of a gap whose denominator moves. And because several
results here are nulls, we read non-significance off the minimum detectable effect rather
than the *p*-value — two structurally identical size comparisons in this paper carry
opposite evidential weight for that reason (§3.1, §5.3).
+ Two things follow for how we report. A delayed model's endpoint is strongly seed-dependent
while its early-exposure anchor is not, so we report absolute endpoints beside their nulls
rather than recovery as a percentage of a gap whose denominator moves. And because several
results here are nulls, we read non-significance off the minimum detectable effect rather
than the *p*-value: two structurally identical size comparisons in this paper carry
opposite evidential weight for that reason (§3.1, §5.3).
```

### REVIEW §2 Methods — `draft-sections-1-2.md`:60
*UNCLASSIFIED -> colon*

```
- All models are decoder-only transformers trained from random initialization; **no pretrained
checkpoint is used anywhere in this work, including for baselines.** This is the design
constraint that makes the rest of the paper meaningful — a model that has already passed
through an uncontrolled pretraining history cannot be asked when it first met a skill.
+ All models are decoder-only transformers trained from random initialization; **no pretrained
checkpoint is used anywhere in this work, including for baselines.** This is the design
constraint that makes the rest of the paper meaningful: a model that has already passed
through an uncontrolled pretraining history cannot be asked when it first met a skill.
```

### §2 Methods — `draft-sections-1-2.md`:74
*paired -> parentheses*

```
- Both columns are given because untied embeddings make them tell different stories. At
vocabulary 16,384 the input embedding and output projection are 4.19M parameters each at the
smallest size — **72.7 % of that model, against 34.8 % at the largest** — so the three
configurations span 5.2× in total parameters but 12.5× in the parameters that do computation.
Size comparisons in this paper are stated in total parameters, which is the axis our
pre-registered analyses used; readers translating our scale results into non-embedding terms
should note that the effective range is wider than the labels suggest, and that the smallest
model is unusually embedding-dominated (§2.5).
+ Both columns are given because untied embeddings make them tell different stories. At
vocabulary 16,384 the input embedding and output projection are 4.19M parameters each at the
smallest size (**72.7 % of that model, against 34.8 % at the largest**), so the three
configurations span 5.2× in total parameters but 12.5× in the parameters that do computation.
Size comparisons in this paper are stated in total parameters, which is the axis our
pre-registered analyses used; readers translating our scale results into non-embedding terms
should note that the effective range is wider than the labels suggest, and that the smallest
model is unusually embedding-dominated (§2.5).
```

### REVIEW §2 Methods — `draft-sections-1-2.md`:83
*UNCLASSIFIED -> colon*

```
- Context length is 512 tokens. Positional information is rotary (θ = 10,000); normalization is
pre-norm; activations are GELU. **Input embeddings and the output projection are untied** —
a deliberate choice, since one of our interventions resets the embedding layer and tied
weights would confound an input-representation reset with a change to the prediction head.
Attention is `F.scaled_dot_product_attention`, which dispatches to FlashAttention-2 kernels
on our GPU. Training uses bf16 autocast, micro-batches of 8 × 512 tokens with 4 gradient
accumulation steps, AdamW (β = 0.9/0.95, weight decay 0.1, gradient clipping 1.0), a peak
learning rate of 6e-4 with 300 warmup steps and cosine decay to 0.1 × peak. `torch.compile`
is available in the trainer but was **disabled for the runs reported here**. Every run
reported here is 5,000 optimizer steps over 81.92M tokens. Checkpoint spacing varies by run
family (100 steps for the 23M arithmetic runs, 1,250 at 11M and 60M, 2,500 for the size
sweep); it sets how finely a trajectory can be re-read afterwards and nothing else, since
delay and intervention triggers fire on exact step count.
+ Context length is 512 tokens. Positional information is rotary (θ = 10,000); normalization is
pre-norm; activations are GELU. **Input embeddings and the output projection are untied**: a deliberate choice, since one of our interventions resets the embedding layer and tied
weights would confound an input-representation reset with a change to the prediction head.
Attention is `F.scaled_dot_product_attention`, which dispatches to FlashAttention-2 kernels
on our GPU. Training uses bf16 autocast, micro-batches of 8 × 512 tokens with 4 gradient
accumulation steps, AdamW (β = 0.9/0.95, weight decay 0.1, gradient clipping 1.0), a peak
learning rate of 6e-4 with 300 warmup steps and cosine decay to 0.1 × peak. `torch.compile`
is available in the trainer but was **disabled for the runs reported here**. Every run
reported here is 5,000 optimizer steps over 81.92M tokens. Checkpoint spacing varies by run
family (100 steps for the 23M arithmetic runs, 1,250 at 11M and 60M, 2,500 for the size
sweep); it sets how finely a trajectory can be re-read afterwards and nothing else, since
delay and intervention triggers fire on exact step count.
```

### §2 Methods — `draft-sections-1-2.md`:107
*relative clause*

```
- Each run is identified by a `run_id` encoding skill, delay, exposure fraction, size and seed;
checkpoints store optimizer state and a metadata dictionary alongside weights. **A `run_id`
is the identity of a trained model, not a label for a configuration** — where two runs share
a configuration but were trained separately, we treat them as different models and never
average or interchange their numbers.
+ Each run is identified by a `run_id` encoding skill, delay, exposure fraction, size and seed;
checkpoints store optimizer state and a metadata dictionary alongside weights. **A `run_id`
is the identity of a trained model, not a label for a configuration**, where two runs share
a configuration but were trained separately, we treat them as different models and never
average or interchange their numbers.
```

### §2 Methods — `draft-sections-1-2.md`:122
*paired -> parentheses*

```
- The syntax metric requires a word of justification, because the obvious one fails. Raw
BLiMP-style *wh*-vs-*that* pairs differ by a single word, so their score is dominated by the
unigram frequency prior for "that": an untrained-for-syntax model scores 0.000 on one
paradigm and 1.000 on its complement while knowing nothing about gap licensing. Taking the
interaction of the with-gap and no-gap conditions cancels that main effect — a purely
frequency-driven model lands at exactly 0.500 — so only genuine gap-licensing knowledge moves
the score. All syntax numbers in this paper are that interaction, and 0.500 is its floor.
+ The syntax metric requires a word of justification, because the obvious one fails. Raw
BLiMP-style *wh*-vs-*that* pairs differ by a single word, so their score is dominated by the
unigram frequency prior for "that": an untrained-for-syntax model scores 0.000 on one
paradigm and 1.000 on its complement while knowing nothing about gap licensing. Taking the
interaction of the with-gap and no-gap conditions cancels that main effect (a purely
frequency-driven model lands at exactly 0.500), so only genuine gap-licensing knowledge moves
the score. All syntax numbers in this paper are that interaction, and 0.500 is its floor.
```

### §2 Methods — `draft-sections-1-2.md`:151
*paired -> parentheses*

```
- We pre-registered three analyses: a mixed-effects model of final accuracy on delay × size, a
factorial model of recovery percentage on intervention × skill × delay, and a model of
circuit overlap on intervention × skill. None is estimable on the design that survived. After
the floor exclusion the first has twelve observations with one per seed × cell and no
within-cell replication for a random effect. The second has a banned denominator — recovery
percentage divides by the seed-variable null — and two of its three factors are constants.
The third has both factors constant. We replaced them with per-seed paired tests, an
absolute-endpoint gate, and a rank correlation with exact permutation *p*, respectively, and
say so here rather than presenting the replacements as the original plan.
+ We pre-registered three analyses: a mixed-effects model of final accuracy on delay × size, a
factorial model of recovery percentage on intervention × skill × delay, and a model of
circuit overlap on intervention × skill. None is estimable on the design that survived. After
the floor exclusion the first has twelve observations with one per seed × cell and no
within-cell replication for a random effect. The second has a banned denominator (recovery
percentage divides by the seed-variable null), and two of its three factors are constants.
The third has both factors constant. We replaced them with per-seed paired tests, an
absolute-endpoint gate, and a rank correlation with exact permutation *p*, respectively, and
say so here rather than presenting the replacements as the original plan.
```

### §3 Detect — `draft-sections-3-4.md`:17
*paired -> parentheses*

```
- Every model in this section sees the same number of arithmetic tokens; the conditions
differ only in *when* those tokens enter the stream. The delayed condition (d50) begins
arithmetic exposure at the midpoint of training, the early condition (d0) at step 0, and
both are read at step 5000 on the same 500 held-out problems. Because seeds are crossed —
the same seed gives the same initialization in both conditions — the design-appropriate
statistic is the per-seed paired deficit, tested separately at each size and Holm-corrected
across the two confirmatory sizes.
+ Every model in this section sees the same number of arithmetic tokens; the conditions
differ only in *when* those tokens enter the stream. The delayed condition (d50) begins
arithmetic exposure at the midpoint of training, the early condition (d0) at step 0, and
both are read at step 5000 on the same 500 held-out problems. Because seeds are crossed (the same seed gives the same initialization in both conditions) the design-appropriate
statistic is the per-seed paired deficit, tested separately at each size and Holm-corrected
across the two confirmatory sizes.
```

### §3 Detect — `draft-sections-3-4.md`:45
*paired -> parentheses*

```
- The deficit is of **similar magnitude at both sizes**: the 60M–23M difference is 0.015,
Welch *p* = 0.72, with a minimum detectable effect at 80 % power of **0.144**. That null
constrains — it excludes any size effect larger than about two-thirds of the deficit itself
— but it is not a claim that size is irrelevant, and it should not be read as one. We
return to this in §5, where a structurally identical size comparison carries the opposite
evidential weight for one reason only: its MDE exceeds the entire range its outcome
occupies. The two nulls are best read as a pair.
+ The deficit is of **similar magnitude at both sizes**: the 60M–23M difference is 0.015,
Welch *p* = 0.72, with a minimum detectable effect at 80 % power of **0.144**. That null
constrains (it excludes any size effect larger than about two-thirds of the deficit itself), but it is not a claim that size is irrelevant, and it should not be read as one. We
return to this in §5, where a structurally identical size comparison carries the opposite
evidential weight for one reason only: its MDE exceeds the entire range its outcome
occupies. The two nulls are best read as a pair.
```

### §3 Detect — `draft-sections-3-4.md`:53
*paired -> parentheses*

```
- **The severity of the deficit is seed-variable, and the variability lives on the delayed
side.** The early anchor is tight across seeds (SD 0.007 at 23M) while the delayed
endpoint is not (SD 0.051), so per-seed deficits range from 0.166 to 0.274 at 23M against a
baseline that barely moves. Seed 1 is the deepest-hit on both readings — lowest delayed
endpoint and largest deficit — and reproduces as such when retrained from scratch. This is
why every recovery result in §4 is reported as an absolute endpoint beside its null rather
than as a percentage of the gap: the gap is a seed-variable denominator, so a
recovery-percentage is unstable by construction even when the underlying effect is not.
+ **The severity of the deficit is seed-variable, and the variability lives on the delayed
side.** The early anchor is tight across seeds (SD 0.007 at 23M) while the delayed
endpoint is not (SD 0.051), so per-seed deficits range from 0.166 to 0.274 at 23M against a
baseline that barely moves. Seed 1 is the deepest-hit on both readings (lowest delayed
endpoint and largest deficit), and reproduces as such when retrained from scratch. This is
why every recovery result in §4 is reported as an absolute endpoint beside its null rather
than as a percentage of the gap: the gap is a seed-variable denominator, so a
recovery-percentage is unstable by construction even when the underlying effect is not.
```

### REVIEW §3 Detect — `draft-sections-3-4.md`:81
*UNCLASSIFIED -> colon, coordinating conjunction, paired -> parentheses*

```
- 1. **Endpoint by delay** (*n* = 3): no gap, against a delay-0 baseline that itself ranges
   0.520–0.586 over eight seeds — a spread exceeding any candidate effect — on the same
   harness that resolves arithmetic's 0.214.
2. **Acquisition dynamics**: Welch *p* = 0.46, underpowered alone and carried by probe 3.
3. **Rate / bimodality**: a pre-registered compound gate over **72 runs** (24 seeds × 3
   sizes) returns DISCONFIRM — the mean rises monotonically with size (0.528 → 0.538 →
   0.552) but fails its floor at threshold 0.56 (top size 0.208 < 0.40) and its trend at
   0.57 (*p* = 0.074), passing only at the permissive 0.55 — and this powered test, sized in
   advance by simulation, is why long-distance syntax left the main grid.
4. **Representational probe**: delay-0 versus delay-50 *p* = 0.48 *while the same probe
   detects the structure at selectivity 0.60, p < 1e-8* — the positive control passes.
5. **Learning-rate budget** (hilr50, *n* = 3): ΔLD = +0.011 ± 0.005, with Δlocal exceeding
   ΔLD at every seed and validation perplexity improving — a non-specific whole-model lift,
   not a rescue of the delayed skill.
+ 1. **Endpoint by delay** (*n* = 3): no gap, against a delay-0 baseline that itself ranges
   0.520–0.586 over eight seeds (a spread exceeding any candidate effect) on the same
   harness that resolves arithmetic's 0.214.
2. **Acquisition dynamics**: Welch *p* = 0.46, underpowered alone and carried by probe 3.
3. **Rate / bimodality**: a pre-registered compound gate over **72 runs** (24 seeds × 3
   sizes) returns DISCONFIRM: the mean rises monotonically with size (0.528 → 0.538 →
   0.552) but fails its floor at threshold 0.56 (top size 0.208 < 0.40) and its trend at
   0.57 (*p* = 0.074), passing only at the permissive 0.55, and this powered test, sized in
   advance by simulation, is why long-distance syntax left the main grid.
4. **Representational probe**: delay-0 versus delay-50 *p* = 0.48 *while the same probe
   detects the structure at selectivity 0.60, p < 1e-8*: the positive control passes.
5. **Learning-rate budget** (hilr50, *n* = 3): ΔLD = +0.011 ± 0.005, with Δlocal exceeding
   ΔLD at every seed and validation perplexity improving: a non-specific whole-model lift,
   not a rescue of the delayed skill.
```

### §3 Detect — `draft-sections-3-4.md`:96
*coordinating conjunction*

```
- Probe 4 is the load-bearing one, for a reason that the other four cannot supply. Probes 1,
2, 3 and 5 all ask whether a critical period is visible *in the behaviour that was
acquired*. Probe 4 asks whether the underlying structure is encoded at all, and finds that
it is — and that its encoding does not depend on when exposure began. **The structure is
learned; what is timing-invariant is the readout.**
+ Probe 4 is the load-bearing one, for a reason that the other four cannot supply. Probes 1,
2, 3 and 5 all ask whether a critical period is visible *in the behaviour that was
acquired*. Probe 4 asks whether the underlying structure is encoded at all, and finds that
it is, and that its encoding does not depend on when exposure began. **The structure is
learned; what is timing-invariant is the readout.**
```

### REVIEW §3 Detect — `draft-sections-3-4.md`:107
*UNCLASSIFIED -> colon*

```
- Under one design, one pipeline and matched total exposure, delayed exposure costs
arithmetic 0.214 ± 0.055 and costs long-distance *wh*-licensing nothing measurable across
five probes. That contrast is the paper's central detection-side result, and it inverts our
pre-registered prediction that syntax would show the *stronger* critical period — a
prediction we restate with its outcome as a standalone item in §7.2, since a reader holding
our pre-registration will look for it there rather than here.
+ Under one design, one pipeline and matched total exposure, delayed exposure costs
arithmetic 0.214 ± 0.055 and costs long-distance *wh*-licensing nothing measurable across
five probes. That contrast is the paper's central detection-side result, and it inverts our
pre-registered prediction that syntax would show the *stronger* critical period: a
prediction we restate with its outcome as a standalone item in §7.2, since a reader holding
our pre-registration will look for it there rather than here.
```

### §3 Detect — `draft-sections-3-4.md`:129
*paired -> parentheses*

```
- The long-distance design's MDE **exceeds its entire headroom by 1.7×**. A total collapse of
the skill to chance would be a change of 0.0283 — smaller than the smallest effect the
design could resolve — and a susceptibility proportionally identical to arithmetic's would
require *n* ≈ 67 per group. The behavioural null therefore cannot, on its own, separate
*syntax is less susceptible* from *syntax had no room to show susceptibility*.
+ The long-distance design's MDE **exceeds its entire headroom by 1.7×**. A total collapse of
the skill to chance would be a change of 0.0283 (smaller than the smallest effect the
design could resolve), and a susceptibility proportionally identical to arithmetic's would
require *n* ≈ 67 per group. The behavioural null therefore cannot, on its own, separate
*syntax is less susceptible* from *syntax had no room to show susceptibility*.
```

### §3 Detect — `draft-sections-3-4.md`:143
*paired -> parentheses*

```
- The differential-susceptibility reading remains open, and its cost is specific rather than
unknown. Under the mean-endpoint readout it is unmeasurable at the *n* achieved here. Under
a rate readout it is **un-run**: the exceedance machinery was only ever pointed at the
delay-0 question, and the seed inventory is asymmetric (24 delay-0 seeds per size, 3 delayed
seeds at 23M alone). A rate-collapse test at 60M — 14 delayed seeds against the existing
24-seed baseline — would settle it; at 23M the same test needs 82 and is out of reach (§8).
+ The differential-susceptibility reading remains open, and its cost is specific rather than
unknown. Under the mean-endpoint readout it is unmeasurable at the *n* achieved here. Under
a rate readout it is **un-run**: the exceedance machinery was only ever pointed at the
delay-0 question, and the seed inventory is asymmetric (24 delay-0 seeds per size, 3 delayed
seeds at 23M alone). A rate-collapse test at 60M (14 delayed seeds against the existing
24-seed baseline) would settle it; at 23M the same test needs 82 and is out of reach (§8).
```

### REVIEW §3 Detect — `draft-sections-3-4.md`:150
*UNCLASSIFIED -> colon*

```
- Finally, a note on what this result is not. Our pre-registration anticipated a
*reopening*-side dissociation: some skill/intervention combinations recover and others do
not. What the data support is a *detection*-side one — syntax never showed a deficit, so no
syntax reopening was attempted. The two are superficially the same finding and are not the
same claim, and we report the one we have.
+ Finally, a note on what this result is not. Our pre-registration anticipated a
*reopening*-side dissociation: some skill/intervention combinations recover and others do
not. What the data support is a *detection*-side one: syntax never showed a deficit, so no
syntax reopening was attempted. The two are superficially the same finding and are not the
same claim, and we report the one we have.
```

### §4 Reopen — `draft-sections-3-4.md`:178
*paired -> parentheses*

```
- All intervention arms fork from a single saved checkpoint at step 2500 — identical weights,
identical optimizer state, identical step — so that arms differ only in the intervention
applied. The `lr_shift` arm restores the learning-rate schedule that a delay-0 model would
have been on at the point delayed exposure begins, then anneals normally. Success was
pre-registered as a two-part condition: the treated endpoint must reach 90 % of that run's
*own* early-exposure anchor, and general capability must not degrade beyond a matched
threshold.
+ All intervention arms fork from a single saved checkpoint at step 2500 (identical weights,
identical optimizer state, identical step), so that arms differ only in the intervention
applied. The `lr_shift` arm restores the learning-rate schedule that a delay-0 model would
have been on at the point delayed exposure begins, then anneals normally. Success was
pre-registered as a two-part condition: the treated endpoint must reach 90 % of that run's
*own* early-exposure anchor, and general capability must not degrade beyond a matched
threshold.
```

### §4 Reopen — `draft-sections-3-4.md`:212
*paired -> parentheses*

```
- Perplexity is reported as the three-number decomposition — early anchor *a*, delayed null
*b*, treated arm *c* — because a single delta cannot distinguish a treated arm that
improved from a null that degraded.
+ Perplexity is reported as the three-number decomposition (early anchor *a*, delayed null
*b*, treated arm *c*), because a single delta cannot distinguish a treated arm that
improved from a null that degraded.
```

### §4 Reopen — `draft-sections-3-4.md`:220
*paired -> parentheses*

```
- At **60M the gate is tripped in all three seeds**: *c* − *b* = +1.670 ± 0.104 ppl. We report
it as a trip and honour it as written. Two controls bound what it can mean. A no-op fork
reproduces the null on both perplexity and accuracy (Δppl −0.012, Δacc +0.011), so fork
mechanics contribute nothing. And at **matched learning rate** the treated arm *leads* the
null by 2.478 ± 0.240 ppl — a lead larger than the breach — because the arm is read
mid-anneal at roughly six times the null's terminal learning rate, with about 4.15 ppl of
annealing gain still unrealized. The breach persists only if the arm's remaining anneal
delivers less than 1.67 ppl of that. The evidence leans toward near-full erasure, but the
anneal slope is unmeasured, so we quote **no point estimate**: the residual cost is bounded
above by 1.67 ppl and a partly real cost is not excluded. Extending the arm past step 5000
is the run that would collapse this bound in either direction (§8).
+ At **60M the gate is tripped in all three seeds**: *c* − *b* = +1.670 ± 0.104 ppl. We report
it as a trip and honour it as written. Two controls bound what it can mean. A no-op fork
reproduces the null on both perplexity and accuracy (Δppl −0.012, Δacc +0.011), so fork
mechanics contribute nothing. And at **matched learning rate** the treated arm *leads* the
null by 2.478 ± 0.240 ppl (a lead larger than the breach), because the arm is read
mid-anneal at roughly six times the null's terminal learning rate, with about 4.15 ppl of
annealing gain still unrealized. The breach persists only if the arm's remaining anneal
delivers less than 1.67 ppl of that. The evidence leans toward near-full erasure, but the
anneal slope is unmeasured, so we quote **no point estimate**: the residual cost is bounded
above by 1.67 ppl and a partly real cost is not excluded. Extending the arm past step 5000
is the run that would collapse this bound in either direction (§8).
```

### §4 Reopen — `draft-sections-3-4.md`:234
*paired -> parentheses*

```
- The same intervention applied to delayed *syntax* models — replicated at all three seeds,
matching the depth of the arithmetic arm — moves nothing: Δ local = −0.024 / −0.004 /
+0.044 and Δ LD = −0.021 / +0.007 / +0.015, sign-inconsistent around zero at every seed.
A single-seed control could not have distinguished "no generic effect" from "an effect that
happened to land near zero once", which is why the control was run at the arm's own seed
depth. Whatever `lr_shift` does, it is not a generic late-training lift.
+ The same intervention applied to delayed *syntax* models (replicated at all three seeds,
matching the depth of the arithmetic arm) moves nothing: Δ local = −0.024 / −0.004 /
+0.044 and Δ LD = −0.021 / +0.007 / +0.015, sign-inconsistent around zero at every seed.
A single-seed control could not have distinguished "no generic effect" from "an effect that
happened to land near zero once", which is why the control was run at the arm's own seed
depth. Whatever `lr_shift` does, it is not a generic late-training lift.
```

### REVIEW §4 Reopen — `draft-sections-3-4.md`:253
*UNCLASSIFIED -> colon*

```
- | **freeze** | **0.332** | **harmful** — below the entire null range |
+ | **freeze** | **0.332** | **harmful**: below the entire null range |
```

### REVIEW §4 Reopen — `draft-sections-3-4.md`:261
*UNCLASSIFIED -> colon*

```
- The one arm we extended to *n* = 3 decides the mechanism. Concentrated re-exposure alone
leaves the endpoint at 0.574 — at the null. Combined with the schedule shift it reaches
0.781 ± 0.031, far above cram alone and indistinguishable from the shift by itself: the
per-seed differences are −0.006 / +0.016 / +0.032, straddling zero with no consistent sign.
(We report the signs rather than the +0.014 mean, because a near-zero mean with disagreeing
signs is an absence of effect, not a small effect obscured by pooling.)
+ The one arm we extended to *n* = 3 decides the mechanism. Concentrated re-exposure alone
leaves the endpoint at 0.574: at the null. Combined with the schedule shift it reaches
0.781 ± 0.031, far above cram alone and indistinguishable from the shift by itself: the
per-seed differences are −0.006 / +0.016 / +0.032, straddling zero with no consistent sign.
(We report the signs rather than the +0.014 mean, because a near-zero mean with disagreeing
signs is an absence of effect, not a small effect obscured by pooling.)
```

### REVIEW §4 Reopen — `draft-sections-3-4.md`:274
*UNCLASSIFIED -> colon*

```
- A final arm belongs in the record without supporting a claim. The 11M `lr_shift` arm passes
its gate (0.747 against a gate of 0.484) but the pass is degenerate: the gate is 90 % of an
anchor that failed the §3.1 floor, and the arm *exceeds its own early anchor* (0.538),
which indicates that the 11M anchor was learning-rate-starved rather than that reopening
over-recovers. The 11M arm enters the paper only through its mechanism decomposition, where
the delayed lift of +0.389 splits into a generic component (+0.115) and an
exposure-alignment component (+0.274) — about 70 % of the effect requiring the delayed
exposure profile. That decomposition is exploratory and carries no headline.
+ A final arm belongs in the record without supporting a claim. The 11M `lr_shift` arm passes
its gate (0.747 against a gate of 0.484) but the pass is degenerate: the gate is 90 % of an
anchor that failed the §3.1 floor, and the arm *exceeds its own early anchor* (0.538),
which indicates that the 11M anchor was learning-rate-starved rather than that reopening
over-recovers. The 11M arm enters the paper only through its mechanism decomposition, where
the delayed lift of +0.389 splits into a generic component (+0.115) and an
exposure-alignment component (+0.274): about 70 % of the effect requiring the delayed
exposure profile. That decomposition is exploratory and carries no headline.
```

### §7–§9 — `draft-sections-7-9.md`:14
*participial phrase*

```
- This paper reports two findings about timing, and they operate at different levels. The
first says that *which* skills a late start damages is skill-specific: arithmetic pays a
persistent cost for a delayed start, long-distance *wh*-licensing pays nothing we can
measure. The second says that where the damage does occur, it is carried by the optimizer's
learning-rate schedule rather than by representations that have hardened against revision —
restoring the schedule restores the skill, and adding skill data without restoring the
schedule does not. A late start hurts some skills and not others; when it hurts, what has
closed is the model's step size, not its mind.
+ This paper reports two findings about timing, and they operate at different levels. The
first says that *which* skills a late start damages is skill-specific: arithmetic pays a
persistent cost for a delayed start, long-distance *wh*-licensing pays nothing we can
measure. The second says that where the damage does occur, it is carried by the optimizer's
learning-rate schedule rather than by representations that have hardened against revision, restoring the schedule restores the skill, and adding skill data without restoring the
schedule does not. A late start hurts some skills and not others; when it hurts, what has
closed is the model's step size, not its mind.
```

### §7–§9 — `draft-sections-7-9.md`:25
*paired -> parentheses*

```
- **Prediction #1 — that syntax would show the *stronger* critical period — was inverted.**
Arithmetic carries the effect at both confirmatory sizes; long-distance *wh*-licensing shows
no detectable deficit under five distinct probes. We had reasoned that long-distance
dependencies, being scaffolded by earlier-acquired structure, would be the more fragile to a
late start. The opposite skill turned out to be the fragile one, and this inversion is the
origin of the dissociation in §3.3 rather than an incidental miss.
+ **Prediction #1 (that syntax would show the *stronger* critical period) was inverted.**
Arithmetic carries the effect at both confirmatory sizes; long-distance *wh*-licensing shows
no detectable deficit under five distinct probes. We had reasoned that long-distance
dependencies, being scaffolded by earlier-acquired structure, would be the more fragile to a
late start. The opposite skill turned out to be the fragile one, and this inversion is the
origin of the dissociation in §3.3 rather than an incidental miss.
```

### §7–§9 — `draft-sections-7-9.md`:32
*paired -> parentheses*

```
- **Prediction #2 — that embedding reset would be the most effective intervention — was not
borne out.** Reset produced no detectable recovery, landing at 0.586 against a null whose
cross-seed range is [0.524, 0.626]; the learning-rate shift was the only arm to clear the
pre-registered gate. Because reset was run at a single seed, this is a statement about what
we could detect, not a demonstration that reset fails: one seed cannot be separated from a
null that spans 0.10. The prediction's second half — that reset would carry the largest
forgetting risk — was also not observed as stated.
+ **Prediction #2 (that embedding reset would be the most effective intervention) was not
borne out.** Reset produced no detectable recovery, landing at 0.586 against a null whose
cross-seed range is [0.524, 0.626]; the learning-rate shift was the only arm to clear the
pre-registered gate. Because reset was run at a single seed, this is a statement about what
we could detect, not a demonstration that reset fails: one seed cannot be separated from a
null that spans 0.10. The prediction's second half (that reset would carry the largest
forgetting risk) was also not observed as stated.
```

### §7–§9 — `draft-sections-7-9.md`:40
*paired -> parentheses*

```
- **Prediction #3 — that a reset-driven recovery would build new circuitry — was never
testable, because its antecedent did not occur.** The prediction was conditional on reset
being the winning intervention. Reset never produced recovery, so the conditional was never
entered. §5 diagnoses models recovered by the learning-rate shift, which prediction #3 did
not cover; those results should not be read as confirming or disconfirming it. We state this
explicitly because §5 occupies the position in the paper where a reader holding our
pre-registration will look for #3's answer.
+ **Prediction #3 (that a reset-driven recovery would build new circuitry) was never
testable, because its antecedent did not occur.** The prediction was conditional on reset
being the winning intervention. Reset never produced recovery, so the conditional was never
entered. §5 diagnoses models recovered by the learning-rate shift, which prediction #3 did
not cover; those results should not be read as confirming or disconfirming it. We state this
explicitly because §5 occupies the position in the paper where a reader holding our
pre-registration will look for #3's answer.
```

### §7–§9 — `draft-sections-7-9.md`:96
*coordinating conjunction*

```
- Timing of exposure matters in these models, but not uniformly and not for the reason we
expected. Delayed arithmetic exposure leaves a deficit that further training does not close;
delayed long-distance *wh*-licensing leaves nothing we can detect, which inverts our
prediction and which we state in absolute terms because the two skills are not equally
observable. Where the deficit exists it reopens, and it reopens when the learning-rate
schedule is restored rather than when the missed data is concentrated — so what closes at
this onset behaves like a property of the optimizer rather than of entrenched representation.
Whether the skill contrast reflects genuine differential susceptibility remains open, and we
have named the fourteen-seed experiment that would decide it.
+ Timing of exposure matters in these models, but not uniformly and not for the reason we
expected. Delayed arithmetic exposure leaves a deficit that further training does not close;
delayed long-distance *wh*-licensing leaves nothing we can detect, which inverts our
prediction and which we state in absolute terms because the two skills are not equally
observable. Where the deficit exists it reopens, and it reopens when the learning-rate
schedule is restored rather than when the missed data is concentrated, so what closes at
this onset behaves like a property of the optimizer rather than of entrenched representation.
Whether the skill contrast reflects genuine differential susceptibility remains open, and we
have named the fourteen-seed experiment that would decide it.
```

