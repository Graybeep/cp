# Does English-token exposure explain the cross-mixture attraction values?

**Ran 2026-07-27. Re-analysis only — zero training, zero new evaluation.**
Artifacts: `scripts/agreement_exposure_recheck.py`, `exposure_recheck.json`.

## The hypothesis under test

From `mixture_replication.md` §"A substantive alternative reading": the two mixtures are not
matched on natural-language volume, so if attraction reduction is driven by **English exposure**
rather than **parameter count**, arith-60.3M should behave like an English-starved (effectively
smaller / under-trained) model — which would explain why it shows *more* attraction than
syntax-60.3M rather than less.

Two testable prongs were named: (A) arith-60.3M's English exposure should land closest to the
syntax mixture's 23.2M cell; (B) arith-60.3M's attraction (+0.334) should then sit between
syntax-23.2M (+0.314) and syntax-60.3M (+0.201).

## The exposure ledger kills prong A structurally

From each run's `realized_exposure.json` (verified identical across seeds 1–3 in all six cells):

| | base (BabyLM) | skill stream | **English total** | non-English |
|:--|--:|--:|--:|--:|
| syntax mixture (exp10) | 73,728,000 | 8,192,000 *(wh-question English text)* | **81,920,000** | 0 |
| arith mixture (exp20) | 65,536,000 | 16,384,000 *(synthetic digits)* | **65,536,000** | 16,384,000 |

Total budget is 81,920,000 tokens in both (5,000 steps × 20,000 micro-batches × 4,096 tokens),
and — the decisive fact — **English exposure does not vary with model size at all.** Every run,
at every size, is the same fixed token budget; only the parameter count changes. Within-mixture
variance in English tokens across sizes is **exactly 0**.

So arith-60.3M's English exposure (65.536M) is **equidistant from all three syntax cells**
(16.384M from each of 11.5M, 23.2M, 60.3M, which all sit at 81.92M). There is no "closest"
size — the mapping prong A asks for does not exist.

**Consequence, and this is the whole result:** a variable with zero within-mixture variance can
only ever produce a **uniform offset between the two mixtures — never a size-dependent
divergence.** English exposure predicts a mixture × scale interaction of **exactly zero**. The
entire anomaly being explained *is* an interaction (+0.177, p = 0.050), concentrated at 60.3M.
The hypothesis explains the component that is absent and none of the component that is present.

### The strongest size-varying repair also fails

The only way to make exposure vary with size is a composite — English **tokens per parameter**.
That is direction-ambiguous, so both readings were computed, under both accountings of "English"
(all natural language, ratio 0.800; or diverse base corpus only, ratio 0.889 — the "~10% less"
figure the original note used):

| reading | arith-60.3M maps to a syntax model of | near 23.2M? |
|:--|--:|:--|
| tokens/param causal (÷0.800) | 75.4M | no — *above* 60.3M |
| less data → smaller effective size (×0.800) | 48.2M | no |
| tokens/param, base-only (÷0.889) | 67.8M | no |
| effective size, base-only (×0.889) | 53.6M | no |

Every variant lands between 48M and 75M. None lands near 23.2M. Prong A fails under every
reading, not just the literal one.

## Prong B fails on the numbers

arith-60.3M = **+0.334**, interval [syntax-60.3M +0.201, syntax-23.2M +0.314]. It sits
**+0.020 above the top of the interval**, not inside it.

## The magnitude is off by 4×, and the observed offsets are not uniform

Fitting the syntax scale line (−0.3161/decade ± 0.0281, n = 72), the **largest** uniform offset
any exposure variant can produce is **|0.031|** (|0.016| under the base-only accounting). The
60.3M divergence is **+0.133 — 4.3× that ceiling.** For exposure alone to produce +0.133, the
arith models would need 0.379× the syntax mixture's English (or 2.64×, under the other reading).
They have 0.800×.

And the observed per-size offsets are not a uniform offset in the first place — they **change
sign**: +0.018 (11.5M), −0.054 (23.2M), +0.133 (60.3M). A constant cannot fit that shape.
(For completeness: the pooled offset is +0.033 ± 0.038 vs a predicted −0.031, t = 1.64,
p = 0.10 — not individually rejected, but the pooled offset is the wrong quantity, since the
pattern it averages over is precisely the non-uniformity at issue.)

## Independent check: the required mechanism is measurably absent

The exposure account requires arith models to be **English-degraded**, and requires that
degradation to be **largest at 60.3M**, where the divergence is largest. `val.bin` is the same
held-out English for every run in the project, so this is directly measurable — no new eval:

| size | arith-mix ppl | syntax-mix ppl | English penalty | Welch p |
|:--|--:|--:|--:|--:|
| 11.5M | 59.75 | 57.07 | **+2.68** (+4.7%) | 4.1e-05 |
| 23.2M | 48.55 | 46.82 | +1.73 (+3.7%) | 0.22 |
| 60.3M | **41.48** | 41.60 | **−0.12 (−0.3%)** | 0.61 |

The 20%-less-English penalty **shrinks monotonically with scale and is gone by 60.3M** — where
the arith models are statistically tied with, and numerically ahead of, the syntax models on
held-out English. arith-60.3M has the best English perplexity of all six cells.

Correlating the per-size English penalty against the per-size attraction divergence gives
**r = −0.74**. The exposure account requires this to be *positive* (bigger English handicap →
bigger divergence). It is negative: the mixtures are **best matched on English competence at
exactly the size where attraction diverges most.**

## Verdict

**The exposure account is refuted, cleanly and on four independent grounds:** the mapping it
needs doesn't exist (zero within-mixture variance), it predicts a zero interaction where the
whole effect is an interaction, its magnitude ceiling is 4.3× too small, and the English
degradation it requires is absent at the one size that matters.

It resolved cleanly — but in the *negative*. That eliminates the leading substantive confound;
it does **not** establish the scale claim. The 60.3M divergence is now a bare anomaly with two
remaining candidates:

1. **n = 3 sampling noise.** Realized power for the observed 0.114 gap was 0.21.
2. **A genuine mixture × scale interaction** (p = 0.050).

Nothing here discriminates between those, so **the properly-powered replication is still
needed** — arguably more so, since the alternative that would have explained it away without
GPU time is now gone. What *has* changed is the design: the divergence is concentrated in one
cell, so seeds at 60.3M are worth more than seeds spread across all three sizes. See
`power_60M.json`.
