# Draft — §6 Agreement attraction (draft 1, 2026-08-05)

Compressed from `notes/agreement-attraction-writeup.md` (§§1–8, ~40k words) to **≈ 0.65
page** (rebalanced 2026-08-06, compressed to it 2026-08-07 — see the length note at the
end). Numbers owned by `logs/sv_agreement_probe/*.json` +
`replication_60M_results.md`; estimator sensitivity by
`scripts/agreement_slope_imbalance_check.py`.

**Compression rule applied:** label-preserving, not prose-shortening. Every sentence that
survived carries its label's qualifiers with it; material that could not keep its
qualifiers at this length was cut whole rather than trimmed into an unlabelled version
(see the cut list at the end).

---

# §6 A secondary probe: agreement attraction across scale (exploratory except where stated)

**Multiplicity scope for this section, declared here rather than inferred.** One family in
§6 is confirmatory: **F6**, the paired attraction−complexity contrast, one test per mixture,
Holm-corrected at *m* = 2 (the syntax mixture survives; the arithmetic one does not). Every
other *p* reported below — the attraction magnitude at a single size, the three
mismatch-versus-chance tests, the complexity-slope null, the per-size cross-mixture
contrasts, and the within-run trajectory — is **exploratory by declaration**: uncorrected,
descriptive, and carrying no headline. They are reported because withholding them would
misrepresent what was looked at, not because any of them is being asserted. This section
analyses models trained for a different purpose, so its tests were not part of the
pre-registered design and are not promoted by consistency with it.

The models trained for this study support one question the critical-period design did not
set out to ask. Subject–verb agreement is acquired almost perfectly here (0.98 on
controlled items), yet fails when a number-mismatching noun intervenes. A fully-crossed item
design, in which the verb-form frequency prior cancels *exactly*,
separates two components of that failure: **attraction**, the cost of an intervening noun
whose number mismatches the head, and **complexity**, the cost of any intervening material
at all.

**Attraction is real, large, and the dominant component.** At 23.2M it costs
+0.303 ± 0.064 (*p* = 8.6e-05) against complexity's +0.143 ± 0.035, in both number
directions and both frame types. A unigram-frequency control on the same items returns
exactly 0.000 by construction, so the effect cannot be a verb-form prior. Attraction exceeds
complexity at every size and is positive in **81 of 81 models** across two independent
training mixtures: 72 syntax-enriched (3 sizes × 24 seeds) and 9 arithmetic-enriched.

**Whether these models are anti-syntactic is scale-dependent.** Mismatch accuracy on
controlled items is 0.405 at 11.5M (below chance, *p* < 0.001), 0.511 at 23.2M (at chance,
*p* = 0.435), and 0.610 at 60.3M (above chance, *p* < 0.001). Below-chance agreement occurs
only at the smallest size.

**Attraction declines with scale; whether it dissociates from complexity depends on the
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

**Within a single run, attraction may also decline over training, but we do not claim it.**
Between steps 2500 and 5000 the isolated attraction cost falls +0.392 → +0.303 (paired,
*n* = 6 at 23.2M, 5/6 seeds): marginal at *p* = 0.053, with 53 % power against this effect
size and an MDE of 0.125; roughly ten seeds would decide it. The mismatch cell itself
improves significantly over the same interval (+0.070, *p* = 0.041). We report the direction
and the power, and treat the trajectory as suggestive only.

**The standard metric gets this wrong at every scale, in three different ways.** Uncontrolled
BLiMP-style items report below-chance behaviour at all three sizes, which is true only at the
smallest; the frequency-controlled metric locates it correctly; and only the crossed design
separates attraction from complexity. Three metrics, three conclusions, on the same models.
A related trap is worth naming: pooling all 72 models into one attraction estimate averages
across a size-dependent effect and yields a number that describes no size: an
averaging-across-a-dissociation error, not a confound.

This is also why the skill sits here rather than in §3. Agreement is unusable as a
delayed-exposure target: it saturates by step 2500 and its supporting structure appears in
roughly 39 % of the corpus against *wh*-extraction's 1.5 %, so it cannot meaningfully be
withheld. Being *learnable enough to measure* and being *delayable* are different properties,
and a design like ours needs both.

### §6 sentence → claim audit

| passage | claim | label | obligation honoured |
|:--|:--|:--|:--|
| crossed design, exact prior cancellation, unigram control 0.000 | **17** | ESTABLISHED | the control that makes it not-a-frequency-artifact stated with the effect |
| +0.303 vs +0.143; 81/81 across two mixtures | **17** | ESTABLISHED | replication scope explicit |
| 0.405 / 0.511 / 0.610 | **20** | ESTABLISHED | "only at the smallest size", not "below chance" unqualified |
| −0.316/decade, n = 72 | **18** (syntax) | ESTABLISHED | balanced design; estimator-invariant |
| −0.207/decade, p = 0.033 OLS / 0.073 robust, (3,3,8) leverage | **18** (arithmetic) | SUGGESTIVE | both p's quoted, never the OLS alone; imbalance named |
| complexity flat, paired contrast −0.333 vs −0.095 | **19** | ESTABLISHED (single-mixture) | scope in the sentence |
| 60.3M divergence +0.065, p = 0.091, CI contains +0.133 | **19** (replication half) | BOUNDED-NOT-EXCLUDED | banned "did not reproduce / is excluded" avoided verbatim |
| trajectory +0.392 → +0.303, p = 0.053, 53 % power, MDE 0.125 | **21** | SUGGESTIVE (exploratory) | power numbers travel with the claim; "we do not claim it" stated in the sentence |
| exposure alternative excluded | — | — | closed qualification, stated as closed |
| three metrics three conclusions; pooled-72 | — | — | methodological byproduct; named as an averaging error, **not** a confound |
| saturation + 39 % vs 1.5 % corpus share | — | — | links C5 to the main design; explains the placement |

---

## Length note — 2026-08-07

The rebalance's §6 line (0.75 → 0.65 page, "the three-metrics argument compresses; claim 21
stays") was applied: the three-metrics and placement paragraphs were compressed and the
scale paragraph tightened, all label-preserving. **It bought 46 words** — 695 → 649, i.e.
0.87 → 0.81 page. The budget is 0.65. Word-level compression has now run out on this
section; anything further removes a labelled sentence, and the two candidates are named
below. Do not trim them into unlabelled versions to close the gap.

*(The page figures in this paragraph are on the superseded geometry. On the corrected one
— see the ⚠ block in `paper-outline.md` — §6 is 92 lines = 0.90 page against 0.65, i.e.
+0.25. The conclusion is unchanged and in fact strengthened: §6 is still one of the
smallest overruns, and every other section is over too, so paying the paper's overflow
here would still be paying in the wrong currency.)*

## Claim 21's cost, measured 2026-08-09 — do not re-cut it for length

`scripts/move_cost.py` applies the edit to this file and re-wraps the section rather than
counting words, and the answer is that **claim 21 is not where the pages are**:

| what | words out | words back | buys |
|:--|--:|--:|--:|
| cut the trajectory paragraph outright (claim 21 leaves the paper) | 90 | — | **11 lines, 0.11 page** |
| move it to the appendix, body keeps a label-preserving stub | 90 | 48 | **5 lines, 0.05 page** |

The appendix version is the honest one and it buys almost nothing, for the reason the §5
appendix move already demonstrated: an appendix relieves a *narrative* budget, not a
*claims* budget. Claim 21 has a home in the body in the coverage table, and the §6 audit
row requires its power numbers to travel with it — so the stub has to carry *p* = 0.053,
53 % power and MDE 0.125 regardless of where the per-seed values live. A stub that drops
them is not an appendix move; it is the unlabelled version this section's compression rule
forbids.

**So the choice is the real one, unhelped by the appendix: keep claim 21, or delete it.**
Deleting buys 0.11 page against an overflow of 4.56. That is 2 %, for a whole labelled
claim — which is an argument for keeping it, not for cutting it.

## Cut whole (could not keep its qualifiers at 0.65 page)

- ~~The trajectory result~~ — **RESTORED 2026-08-05 as ledger claim 21, SUGGESTIVE
  (exploratory)**, with its power numbers attached in the sentence itself. Costs ~55 words.
  Re-cutting it is one of the two remaining ways to reach 0.65; the other is the placement
  paragraph (~60 words, no claim attached, but it is the only thing tying C5 to the main
  design). **Both are calls to make deliberately, not by trimming.**
- **The per-size cross-mixture contrasts** (exploratory, flagged, no headline).
- **The mechanism discussion** of *why* attraction declines with scale — speculative in the
  source write-up and unlabelled.
- **§5 of the source** (what the standard metric gets wrong) survives only as the
  three-metrics sentence; the full argument is a paper of its own.

## Note for the §7 Discussion

C5 currently has no line in the Discussion. It needs at most one: the same measurement
discipline that produced the §3.2 syntax null — a probe with demonstrated dynamic range
beats an accumulation of agreeing readouts — is what separates attraction from complexity
here. If that line does not appear, the section reads as an unconnected bonus result, which
is the failure mode a compressed section is most exposed to.
