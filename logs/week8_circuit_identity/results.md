# Week 8 — Circuit Identity: did lr_shift REPAIR or REBUILD the arithmetic circuit?

**Question.** The arithmetic critical period (delay-50) is reopened by the LR-shift
intervention (Week 4: 0.52→0.77 exact-match). Diagnose whether the recovered model *repaired
the original circuit* or *built a new one* (CLAUDE.md operational definition).

**Models (per seed, all share seed-init → comparable bases; drift measured by CKA):**
- **A** = `arith_delay0_exp20_size23M_seed{S}` — early baseline (healthy)
- **B** = `arith_delay50_exp20_size23M_seed{S}` — delayed (deficient)
- **C** = `arith_delay50_..._ivlrshift0at2501` — lr_shift recovered

**Method (`scripts/week8_circuit_identity.py`, primitives in `reopen/diagnose.py`):**
1. Ablation localization — mean-ablate each attention head (per-head, on-distribution mean),
   measure exact-match damage → each model's arithmetic circuit = its high-damage head set.
   (Whole-layer mean-ablation *saturates* — any core layer → acc≈0 — so it is non-discriminative;
   documented and localization is done at head granularity.)
2. **Reciprocal ablation transfer (the decisive causal test).** Ablate A's top heads in C, and
   C's own top heads in C. REPAIR ⇒ A's heads damage C too; NEW CIRCUIT ⇒ A's heads spare C
   while C's own heads (a different set) damage it. The C-own-heads arm is the control against
   "C is merely robust to any ablation."
3. Linear CKA of the residual stream per layer (secondary; global geometry).

> ## ⚠ SUPERSEDED IN PART — eval contamination, corrected 2026-07-28
>
> Every number below was computed with an eval set built as
> `ArithConfig(seed=<run seed>)`. Only the **seed-1** draw is held out of training;
> the seed-2/3 draws (at both sizes) contain **19.4 % / 17.4 % trained-on problems**.
> Four of the six triads are affected. All six were re-scored on the held-out draw
> (seed-1 triads reproduce exactly = no-op control). **Read
> `logs/week9_stats/cleaneval_recheck.md` for the corrected numbers.**
>
> What changed: the pooled deficit→transfer relation weakens from ρ = −0.943
> (exact p = 0.0083) to **ρ = −0.714 (exact p = 0.068), LOO-fragile** — no longer
> significant; the §confound collinearity argument **reverses** (corrected deficit
> ranges overlap, means 0.228 vs 0.229, so size *is* separable — but the level
> comparison is **estimable, not answered**: Welch p = 0.39 with MDE 0.809 against a
> 0.681 outcome span, so it excludes nothing [corrected 2026-08-03; this line
> previously read "and is null on level"]); the extrapolation check no longer
> supports anything.
> What survives: the seed-2 vs seed-3 inversion (transfer 0.659 vs 0.868, CIs still
> disjoint on the common draw), 60M reorganizing at every seed, and the
> reproducibility gap (SD 0.349 vs 0.131, 2.7× rather than 5.1×).

## HEADLINE (revised 2026-07-26 after completing 60M seeds 2–3 and patching at all six triads)

> **This section replaces an earlier headline that claimed scale drives reorganization
> independently of deficit depth. That claim is withdrawn — see §confound. The reorganization
> itself replicated; its *attribution to scale* did not survive the added seeds.**

- **Deficit depth is the dominant driver of circuit identity, and the relation is graded rather
  than a coarse two-bucket contrast.** Pooled across all six triads, deeper deficit → lower
  ablation transfer: Spearman **ρ = −0.94, p = 0.005 (n = 6)**, spanning 0.87 (deficit 0.182) to
  0.19 (deficit 0.292). A deficit→transfer line fitted on the 23M triads *alone* predicts the
  three unseen 60M points to within −0.16 / −0.06 / −0.03 (§confound) — one relationship
  accounts for both sizes out of sample.
- **60M reorganizes consistently** — transfer **0.445 / 0.399 / 0.308** (mean 0.384 ± 0.070),
  every seed's bootstrap CI below the 23M repair band, every control ratio ≥ 2.0. The
  pre-registered replication bar (`notes/week8-60M-seed2-prereg.md`) is **met**.
- **Whether scale independently matters is unidentifiable with the current design.** The three
  60M triads all landed at deficits (0.224–0.272) that no 23M triad occupies, so size and deficit
  depth are collinear here (Welch on transfer by size: p = 0.43). Since the extrapolation check
  already shows a single deficit→transfer relation covering both sizes, and deficit is emergent
  from training rather than settable, this is left open rather than pursued (§confound).
- **Activation patching does not discriminate at all** and cannot be used to corroborate any of
  this — flat across all six triads (§patching). The earlier "patching corroborates seed-1
  reorganization" reading was an artifact of running patching on one seed only.
- What *is* new and solid at 60M: the **cross-seed variance collapses** (60M SD 0.070 vs 23M
  SD 0.356). 23M identity is erratic seed-to-seed; 60M is reproducible.

## Result — reciprocal ablation (necessity), n=3 seeds at each of two sizes

Fraction of each model's own accuracy lost when a head set is mean-ablated, and the headline
transfer ratio (how much of A's damage C inherits):

Ordered by deficit depth (not by size) — the ordering that actually organizes the table:

| size·seed | deficit A−B | transfer C/A | 95% CI | control | Jaccard | identity |
|-----:|:-----------:|:------------:|:------:|:-------:|:-------:|:---------|
| 23M·2 | 0.182 (shallowest) | 0.71 | [0.665, 0.757] | 1.4 | 0.09 | repair |
| 23M·3 | 0.206 | 0.87 | [0.834, 0.902] | 1.2 | 0.20 | repair |
| 60M·1 | 0.224 | 0.45 | [0.395, 0.495] | 2.2 | 0.09 | reorganized |
| 60M·2 | 0.250 | 0.40 | [0.342, 0.455] | 2.6 | 0.00 | reorganized |
| 60M·3 | 0.272 | 0.31 | [0.253, 0.363] | 3.4 | 0.09 | reorganized |
| 23M·**1** | **0.292 (deepest)** | **0.19** | — | 5.4 | 0.00 | most reorganized |

*transfer C/A = fraction of C's accuracy lost to ablating A's top heads, ÷ fraction A loses.
control = (C's own-heads damage)/(A's-heads damage on C); >1 means C has its own circuit, so a
low transfer is not mere ablation-robustness. CIs bootstrap over eval problems (eval noise only,
NOT seed variance). C fully recovers at every 60M seed (0.798/0.758/0.762 vs A 0.796/0.834/0.816).*

Group summary — **both numbers shown, never collapsed into a ratio** (seed-depth rule, third clause):

| size | transfer (mean ± SD) | deficit range |
|:--|:--|:--|
| 23M (n=3) | **0.588 ± 0.356** | 0.182 – 0.292 |
| 60M (n=3) | **0.384 ± 0.070** | 0.224 – 0.272 |

Welch on transfer by size: **t = 0.97, p = 0.43** — the size difference in means is *not*
significant, because 23M's spread swamps it. What separates the sizes is **reproducibility**
(SD 0.070 vs 0.356), not level.

### §confound — why the scale claim is withdrawn (`week8_scale_vs_deficit.py`)

The original scale claim compared 60M·1 (deficit 0.224) against "deficit-matched 23M models."
They were not matched: the 23M repair seeds sit at 0.182 and 0.206, both **shallower**. Two
statistics carry the revised reading:

**1. Deficit depth predicts transfer across both sizes pooled: Spearman ρ = −0.94, p = 0.005
(n = 6).** One monotone relation orders all six triads, from 0.19 at deficit 0.292 to 0.87 at
deficit 0.182, with size interleaved rather than separating.

**2. A 23M-only fit extrapolates onto the 60M points.** Fitting transfer ~ deficit on the three
23M triads alone (slope −5.6, r = −0.91; p = 0.275 at n = 3) and extrapolating into the 60M
deficit window predicts **0.60 / 0.46 / 0.33** against observed **0.45 / 0.40 / 0.31** —
residuals −0.16, −0.06, −0.03. The larger models do close to what a 23M model at the same
deficit is predicted to do, using a line fitted without any 60M data. This is the informative
test: it is a genuine out-of-sample check, and the 60M points do not sit systematically off it.

Why size cannot be recovered as a separate factor: **no 23M triad lies in the 60M deficit window
[0.224, 0.272].** The 60M range is nested inside the 23M range but disjoint from every 23M
observation, so size and deficit depth are collinear across these six points. Consistent with
that, Welch on transfer by size gives p = 0.43.[^1]

⇒ **Deficit depth is the dominant driver of circuit identity. Whether scale independently
matters is unidentifiable with the current design.** Not pursued further: separating them would
require triads whose deficit ranges overlap, and deficit is emergent from training rather than
directly controllable, so it could only be attempted by running additional seeds and hoping the
ranges happen to overlap. Given the extrapolation check above already shows one deficit→transfer
relationship accounting for both sizes, the payoff is uncertain and the question is left open
rather than chased.

[^1]: Within-60M Spearman is ρ = −1.00. This is reported for completeness only and is **not
evidence** — at n = 3, any two variables that each vary monotonically across the three points
produce a perfect rank correlation almost mechanically. The pooled n = 6 correlation and the
out-of-sample extrapolation are the statistics that carry weight.

## Result — activation patching (sufficiency) at all six triads

### §gradient — is the seed-2 vs seed-3 difference (0.71 vs 0.87) real? (`week8_transfer_bootstrap.py`)
Bootstrap over eval problems (fixed head sets/models): seed 2 transfer 0.710 **CI [0.665,0.757]**,
seed 3 0.868 **CI [0.834,0.902]** — **disjoint**. So the difference exceeds eval-sampling noise;
these two models genuinely reorganized to different degrees. **But it is NOT a severity gradient:**
the ordering is *inverted* vs deficit (seed 3 has the deeper deficit yet more repair). And the
bootstrap captures only eval variance, **not seed/init variance** — with n=1 model per point it
cannot claim a third shallow-deficit seed lands at 0.87. Conclusion: within the moderate-deficit
regime, degree of repair is real but **not deficit-driven**; only the deep-vs-moderate contrast
(seed 1 vs 2/3) is a robust deficit effect.

### §patching — now run on all six triads (`week8_causal_trace.py`, n=900 each)

Independent causal method (denoising: patch clean head activations into corrupted-operand runs,
measure restoration of the clean answer). Head-level, first-answer-token readout. Ratio below =
(mean restoration of A's top heads in C) ÷ (mean restoration of C's own top heads in C).

| size·seed | deficit | ablation transfer | patching Jaccard | A-heads-in-C | patching ratio |
|:--|:--:|:--:|:--:|:--:|:--:|
| 23M·1 | 0.292 | **0.19** | 0.09 | 0.088 | 0.60 |
| 23M·2 | 0.182 | **0.71** | 0.20 | 0.076 | 0.74 |
| 23M·3 | 0.206 | **0.87** | 0.09 | 0.066 | 0.64 |
| 60M·1 | 0.224 | 0.45 | 0.09 | 0.062 | 0.59 |
| 60M·2 | 0.250 | 0.40 | 0.00 | 0.051 | 0.79 |
| 60M·3 | 0.272 | 0.31 | 0.20 | 0.069 | 0.74 |

**Patching is flat across every triad and corroborates nothing.** Ablation transfer spans
0.19–0.87 (SD 0.256); the patching ratio spans only 0.59–0.79 (SD 0.083). Between-method
agreement is **zero**: Pearson r = 0.045 (p = 0.93), Spearman ρ = −0.09 (p = 0.87). Patching
ratio vs deficit ρ = 0.03 (p = 0.96); by size, Welch p = 0.57.

**This retracts the earlier §patching reading.** With seed 1 alone, "low Jaccard + A's heads
partly sufficient" looked like independent corroboration of seed 1's reorganization. Running the
other five shows patching returns *that same answer for the repair seeds too* — it does not
distinguish a transfer-0.87 repair case from a transfer-0.19 reorganized one. So it never
corroborated the seed-1 verdict; it is simply insensitive here.

**Likely why:** single-head denoising restoration is small for everyone (0.05–0.15 absolute, for
both A's heads *and* C's own), so the ratio is a quotient of two near-floor quantities with little
dynamic range. Mean-ablation, by contrast, moves accuracy across 5–99% of the model's own score.
The cross-method reconciliation ("redundant alternative machinery") is **not established** — it
rests on a contrast patching cannot resolve. Testing sufficiency properly would need head *groups*
patched jointly, not one at a time.

Seed-1 reciprocal table (the most-reorganized case, by ablation), fraction of own acc lost:

```
model          ablate A's-heads   ablate C's-heads
A early              99 %              59 %
B delayed            63 %              99 %
C lr_shift           19 %              99 %      <- A's heads spare C; C's own heads destroy it
```
By ablation, A's top heads {L4H0,L1H7,L4H3,L2H1,L2H7,L4H2} and C's {L3H3,L2H5,L2H2,L2H0,L5H6,L1H5}
share zero members (necessity-based). Patching does **not** independently confirm this (§patching).

### Reading (revised 2026-07-26)
- **Deficit depth is the dominant driver of circuit identity across both sizes**
  (pooled ρ = −0.94, p = 0.005, n = 6; plus the out-of-sample extrapolation check, §confound):
  the deeper the delayed model's deficit, the less the recovered model's arithmetic depends on
  the early baseline's heads. At the deep end (23M·1, 0.292) transfer is 0.19; at the shallow end
  (23M·2, 0.182) it is 0.71.
- **This is now a graded relation, not just a coarse contrast.** The earlier "deep-vs-moderate
  contrast only, NOT a gradient" reading was based on three 23M points where seeds 2/3 invert.
  With six points spanning 0.182–0.292 the monotone trend is clear; the 23M seed-2/3 inversion
  (0.71 vs 0.87, disjoint CIs — §gradient) remains a real local exception to it.
- **Scale: unidentifiable with the current design** (§confound). 60M reorganizes, reproducibly,
  but only at deficits where the 23M-fitted relation predicts roughly the same transfer. Not
  withdrawn as *false* — untestable as specified, and deliberately not chased further.
- **The one clean size effect is variance, not level:** 60M identity is reproducible across seeds
  (SD 0.070), 23M is not (SD 0.356). Worth reporting on its own terms — it mirrors the Week-4
  result that the intervention collapses cross-seed variance in the *endpoint*.
- Pre-registered escalation confirmed (week8-diagnosis-seed-scope): seed 2 ≠ seed 1 ⇒ not
  invariant ⇒ seed 3.

### Methodological note
Method choice decides the answer here. **Mean-ablation (necessity) is the only one of the three
that discriminates.** CKA would have missed the seed-1 rebuild entirely (0.66 — shared init keeps
global geometry similar even where causal wiring differs). Single-head activation patching
(sufficiency) is flat across all six triads and distinguishes nothing (§patching). Whole-layer
ablation saturates. A finding that rests on one of these methods should not be described as
"cross-method validated" until a second method is shown to have *resolving power on the same
contrast* — that check is what overturned the earlier patching corroboration.

## Caveats (remaining)
- **Size and deficit depth are collinear in this dataset** (§confound) — no 23M triad occupies the
  60M deficit window. Any size claim needs triads whose deficit ranges overlap; since deficit is
  training-emergent this is not a targeted run, and the question is left open by decision, not
  by oversight.
- **The 23M-only fit used for the extrapolation check is n=3** (p = 0.275), so the residuals are
  an out-of-sample consistency check, not a calibrated prediction interval.
- **Sufficiency is untested, not resolved.** Single-head patching lacks dynamic range; joint
  head-group patching would be needed to test the "redundant alternative machinery" idea, which is
  currently unsupported rather than supported.
- **Bootstrap CIs capture eval variance only**, not seed/init variance (n=1 model per point).
- **23M seeds 2/3 invert vs deficit** — a real local exception (disjoint CIs) to the pooled trend.

Artifacts: `seed{1,2,3}.json`, `seed{1,2,3}_60M.json`, `transfer_bootstrap_{23M,60M}.json`,
`scale_vs_deficit.json`, `method_agreement.json`, `week8_causal_trace/seed{1,2,3}_{23M,60M}.json`;
scripts `week8_circuit_identity.py`, `week8_causal_trace.py`, `week8_transfer_bootstrap.py`,
`week8_scale_vs_deficit.py`, `week8_method_agreement.py`; primitives `reopen/diagnose.py`;
pre-registration `notes/week8-60M-seed2-prereg.md`.
