# Draft — §7 Discussion, §8 Limitations, §9 Conclusion (draft 1, 2026-08-05)

**Budget conflict RESOLVED 2026-08-06 — rebalance applied.** §1 → 0.60, §3 → 1.85,
§6 → 0.65 released 0.45 page for a standalone **Related Work** section (now between §5 and
§6); §§7–9 hold 0.45 with §7.4 moved out. Total stays 8.00. The note at the end is kept as
the record of what was traded away.

---

# §7 Discussion

## 7.1 Two timing results at different levels

This paper reports two findings about timing, and they operate at different levels. The
first says that *which* skills a late start damages is skill-specific: arithmetic pays a
persistent cost for a delayed start, long-distance *wh*-licensing pays nothing we can
measure. The second says that where the damage does occur, it is carried by the optimizer's
learning-rate schedule rather than by representations that have hardened against revision, restoring the schedule restores the skill, and adding skill data without restoring the
schedule does not. A late start hurts some skills and not others; when it hurts, what has
closed is the model's step size, not its mind.

## 7.2 The three pre-registered predictions

**Prediction #1 (that syntax would show the *stronger* critical period) was inverted.**
Arithmetic carries the effect at both confirmatory sizes; long-distance *wh*-licensing shows
no detectable deficit under five distinct probes. We had reasoned that long-distance
dependencies, being scaffolded by earlier-acquired structure, would be the more fragile to a
late start. The opposite skill turned out to be the fragile one, and this inversion is the
origin of the dissociation in §3.3 rather than an incidental miss.

**Prediction #2 (that embedding reset would be the most effective intervention) was not
borne out.** Reset produced no detectable recovery, landing at 0.586 against a null whose
cross-seed range is [0.524, 0.626]; the learning-rate shift was the only arm to clear the
pre-registered gate. Because reset was run at a single seed, this is a statement about what
we could detect, not a demonstration that reset fails: one seed cannot be separated from a
null that spans 0.10. The prediction's second half (that reset would carry the largest
forgetting risk) was also not observed as stated.

**Prediction #3 (that a reset-driven recovery would build new circuitry) was never
testable, because its antecedent did not occur.** The prediction was conditional on reset
being the winning intervention. Reset never produced recovery, so the conditional was never
entered. §5 diagnoses models recovered by the learning-rate shift, which prediction #3 did
not cover; those results should not be read as confirming or disconfirming it. We state this
explicitly because §5 occupies the position in the paper where a reader holding our
pre-registration will look for #3's answer.

## 7.3 What the mechanism result does not say

Schedule-driven closure is demonstrated at one onset (the midpoint of training), for one
skill, at two model sizes. It is not a claim that representational entrenchment plays no role
in critical periods generally, nor that it plays no role at later onsets in this same setup.
Our intermediate-onset runs exist at a single seed and are not evidence about the shape of
the timing curve.

---

# §8 Limitations

Five specific experiments would move a label, and we name them with their costs rather than
gesturing at future work.

**The dissociation's stronger reading is un-run, not unknowable.** Whether syntax is
genuinely *less susceptible* to delayed exposure, rather than merely having too little
headroom to show susceptibility, is decidable by a rate-collapse test at 60M: 14 delayed
seeds against our existing 24-seed baseline, or 6 at a permissive threshold. This assumes
delay eliminates the high-acquisition mode; if it merely halves it, 59 are needed. The same
test at 23M requires 82 seeds and is out of reach. This is the cheapest outstanding
experiment in the project and the only one that can promote the dissociation past its
current scope.

**The 60M perplexity cost is bounded, not estimated.** Extending the treated arm past step
5,000 to complete its annealing would convert our upper bound of 1.67 ppl into a point
estimate in either direction.

**The circuit-identity gradient is unresolved at *n* = 6.** Additional triads at either size
are the only thing that would settle whether deficit depth grades circuit identity; the
direction persists but four of six leave-one-out deletions break its significance.

**The circuit results are single-method by demonstration.** Joint head-group patching, shown
to have dynamic range on the repair-versus-reorganization contrast, would let those claims
drop the annotation. Three probes we tried do not discriminate.

**One agreement-attraction slope is estimator-dependent.** Three or more further seeds at the
smallest size would rebalance the leverage that makes it significant under OLS and marginal
under a robust standard error.

Beyond these: one corpus, one tokenizer, one language, a maximum of 60.3M parameters, and a
single delay onset carrying the reopening result. Five of six intervention arms are single
seed. The smallest size is excluded by a floor rather than solved.

---

# §9 Conclusion

Timing of exposure matters in these models, but not uniformly and not for the reason we
expected. Delayed arithmetic exposure leaves a deficit that further training does not close;
delayed long-distance *wh*-licensing leaves nothing we can detect, which inverts our
prediction and which we state in absolute terms because the two skills are not equally
observable. Where the deficit exists it reopens, and it reopens when the learning-rate
schedule is restored rather than when the missed data is concentrated, so what closes at
this onset behaves like a property of the optimizer rather than of entrenched representation.
Whether the skill contrast reflects genuine differential susceptibility remains open, and we
have named the fourteen-seed experiment that would decide it.

---

## Drafting notes

- **§7.4 removed 2026-08-06** — positioning moved to a standalone Related Work section
  between §5 and §6, per the applied rebalance. See `draft-related-work.md`. §7 ends at 7.3.
  *(This record sat as an inline parenthetical in §7.3's body until 2026-08-09, where it was
  inside the measured page budget and would have typeset into the paper. Found by
  `cut_dependencies.py`'s completeness check, which flagged it as an unexplained §5
  reference. Same failure as the 2026-08-07 estimator bug: a drafting note placed above a
  section's stop heading is body. Apparatus goes under an apparatus heading.)*

## Redundancy check against Related Work — 2026-08-07

Run before cutting anything here, on the instruction that §7–§9's overrun should not be paid
by deleting something the new Related Work section already says. **It should not be: the
overlap is about 25 words, and in both cases the duplicate copy is Related Work's, not this
section's.**

| overlapping fact | Related Work's use | §7–§9's use | verdict |
|:--|:--|:--|:--|
| reset produced no detectable recovery, *n* = 1 | contrast with the plasticity literature, whose standard remedy is weight perturbation | **§7.2 prediction #2** — accountability against the pre-registration, with the null's range and the "not a demonstration that reset fails" guard | **keep both.** Different jobs; §7.2's is the load-bearing one and is required verbatim by backbone §11. Related Work now cross-references §7.2 rather than re-deriving it |
| three of four circuit probes do not discriminate | positioning against the mech-interp literature; already written as a pointer to §5.1 | **§8** — a costed label-mover ("joint head-group patching would drop the annotation") | **keep both.** Related Work states the result, §8 states its price. Neither is the other |

Two further near-misses that are **not** redundancy: §7.1's two-level timing statement is
adjacent to Related Work's plasticity contrast but makes a different claim (levels of
description, not prior art); §9 restates §7.1 by design, which is what a conclusion is.

**Consequence.** §7–§9's overrun cannot be recovered here. §7.2's three prediction
statements are incompressible by construction — each must name its prediction, its outcome,
and for #2 and #3 why it says less than it appears to — and §8's five entries are the
project's costed-experiment list, the part of a limitations section reviewers actually use.
The next candidate is therefore an **appendix move, not a cut**: §8's seed-count derivations
(the 14 / 59 / 82 arithmetic and its threshold assumptions) could go to supplementary
material while each limitation keeps its named experiment and headline cost in the body.
That is a decision, not an edit, and it is not taken here.

### The move was costed 2026-08-09 — and it is not worth taking

**It buys 3 lines (0.03 page), not the ~0.10 page guessed above.** `scripts/move_cost.py`
applies the edit to this file and re-wraps the section with `page_estimate`'s own measure:
34 words leave, a 9-word appendix pointer comes back, and the section drops from 146 lines
to 143. Cutting the derivations outright rather than moving them buys 4.

Two things were also wrong in the sentence above. The overrun is **+0.99 page**, not +0.61:
the whole page budget was recomputed on 2026-08-09 after an actual compile showed
`page_estimate.py` had the template's font size and leading wrong (see the ⚠ block in
`paper-outline.md`). And ~0.10 page was a word-count guess of exactly the kind that has now
mis-sized three separate moves. At the measured exchange rate of ~8.3 words per line, 34
words could never have bought 0.10 page — that would need ~85.

**Not executed.** A 3-line saving does not justify moving a number out of the body, and §8's
whole value is that its experiments are costed where a reviewer reads them.

---

## Budget note — two problems, one fix (SUPERSEDED 2026-08-07 — see above and `logs/page_estimate.md`)

1. **These three sections need ≈ 0.9 page, not the 0.5 allotted.** §7.2's three prediction
   statements are ~230 words and are not compressible: each must name its prediction and its
   outcome, and #2 and #3 additionally have to state *why* they say less than they appear to.
2. **The outline never allocated a Related Work section.** §7.4 is currently a stub inside
   the Discussion, which is the wrong home for it at this venue — a reviewer will expect
   positioning against loss-of-plasticity and curriculum work, and that argument is where a
   reader decides whether the schedule finding is novel or a rediscovery.

**Proposed rebalance** (total stays at 8.00):

| section | was | now | how |
|:--|--:|--:|:--|
| §1 Introduction | 0.75 | 0.60 | it currently previews results §3–§4 restate in full |
| §3 Detect | 2.00 | 1.85 | 3.2's five-probe list compresses to one sentence per probe |
| §6 Agreement | 0.75 | 0.65 | the three-metrics argument compresses; claim 21 stays |
| **Related Work** | — | **0.45** | new; absorbs §7.4 |
| §7–§9 | 0.50 | 0.45 | with §7.4 moved out, the remainder fits |

The alternative — keeping Related Work inside the Discussion — is defensible at some venues
and saves ~0.15 page of heading overhead, but it puts the novelty argument after all the
results, where a skeptical reader has already formed a view. Flagging as a call rather than
making it.
