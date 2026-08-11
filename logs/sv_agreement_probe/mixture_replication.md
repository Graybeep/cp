# Second-mixture replication of the attraction × scale dissociation

**Ran 2026-07-26.** Evaluation-only, zero training. Artifacts:
`scripts/agreement_mixture_replication.py`, `quartet_scale_arithmix.json`,
`mixture_replication.json`.

## What was tested

The note's headline claim — attraction falls **−0.316/decade** while complexity stays flat
(n=72, p=2.3e-17) — came entirely from **one** training mixture (10% syntax-enriched).
The arithmetic-mixture d0 checkpoints (`arith_delay0_exp20_size{11M,23M,60M}_seed{1,2,3}`,
step 5000) exist, so a second-mixture replication is evaluation-only. Matched to the
original: 400 sextets, step 5000, verb-position scoring, `--skip-sentence`.

**Pre-registered read (fixed before running):**
- **CONFIRM** — attraction slope vs log₁₀(params) negative at p<0.05, ordering 11M>23M>60M,
  complexity slope non-significant.
- **BREAK** — attraction flat/positive, or complexity moves comparably.

## Result: the dissociation does NOT replicate

| | syntax-mix (n=24/size) | arith-mix (n=3/size) |
|:--|:--|:--|
| attraction 11.5M | +0.430 ± 0.071 | +0.448 ± 0.106 |
| attraction 23.2M | +0.314 ± 0.081 | **+0.260 ± 0.036** |
| attraction 60.3M | +0.201 ± 0.056 | **+0.334 ± 0.082** |
| attraction slope | **−0.316/decade, p=2.3e-17** | **−0.139/decade, p=0.278** |
| complexity slope | +0.016/decade, p=0.43 | −0.059/decade, p=0.268 |
| 11M→60M gap | +0.229 [+0.192,+0.266], p=5.6e-16 | +0.114 [−0.106,+0.334], p=0.217 |
| monotone 11>23>60 | **yes** | **no** (23M < 60M) |

Not one pre-registered CONFIRM criterion is met: the slope is half-size and
non-significant, the ordering is non-monotone (60.3M *above* 23.2M), and the
dissociation *pattern* can't even be evaluated because **neither** component moves
significantly in this mixture.

**What DID replicate — the attraction effect itself, emphatically.** +0.347 ± 0.107 across
the 9 arith-mixture models, **positive 9/9 in both number directions**, t=9.72, p=1.05e-05;
unigram control attraction exactly +0.0000. Finding #1 is now confirmed across mixtures.
Only the *scale* claim fails.

## But the test is uninformative, not a refutation

The arith 11M→60M CI is **[−0.106, +0.334]**. It contains **0** *and* the syntax-mixture
**+0.229**. This design cannot distinguish "no scale effect" from "the same scale effect."

**My pre-registered power estimate (~89%) was optimistic and I should have caught it.** It
used the *syntax* mixture's within-size SD (0.063). The arith mixture's is **0.094 — 1.48×
larger**. Realized power at n=3 for a 0.229 gap: **0.62**, not 0.90. For the gap actually
observed (0.114): **0.21**.

- 80% power on a 0.229 gap → **4 seeds/size** (one more seed per size; would need training)
- 80% power on the observed 0.114 gap → **12 seeds/size**

## Marginal evidence the slopes genuinely differ

Pooled regression over all 81 models, attraction ~ log₁₀(params) × mixture:

| term | estimate | p |
|:--|:--|:--|
| log₁₀(params) | −0.3156 ± 0.0295 | 7.4e-17 |
| mixture | +0.0325 ± 0.0261 | 0.22 |
| **log₁₀(params) × mixture** | **+0.1765 ± 0.0886** | **0.050** |

Right at the boundary. Weak evidence the arith-mixture slope is genuinely shallower rather
than a sampling accident — enough to block the claim, not enough to establish an interaction.

Per-size, the divergence is concentrated at the **top**: 11.5M agrees across mixtures
(+0.018, p=0.80), 23.2M is close (−0.054, p=0.10), and **60.3M diverges** (arith +0.334 vs
syntax +0.201, diff +0.133, p=0.098). The syntax mixture's largest models reduce attraction;
the arithmetic mixture's do not.

## A substantive alternative reading — TESTED AND EXCLUDED 2026-07-27

The mixtures are **not matched on natural-language volume**: the arith mixture spends 20% of
its tokens on synthetic arithmetic vs 10% on syntax-enriched *text* for the other. So the
arith models see less English at every size. If attraction reduction is driven by English
exposure rather than parameter count, the arith-60M models should behave like under-trained
large models — which is the observed direction.

**This was re-analysable from existing runs and has now been run** — see
`exposure_recheck.md` / `.json`, `scripts/agreement_exposure_recheck.py`. **It is excluded:**

1. **No mapping exists.** English totals are 81.92M (syntax mix) vs 65.54M (arith mix), and
   both are **held constant across sizes by design** — total budget is 81.92M tokens in every
   run, only parameters vary. Within-mixture variance is exactly zero, so arith-60M is
   equidistant from all three syntax cells. Exposure can produce only a **uniform mixture
   offset, never a size-dependent divergence** — it predicts interaction ≡ 0, and the anomaly
   *is* an interaction.
2. **Magnitude ceiling 4.3× too small.** The largest offset any exposure variant can generate
   under the fitted slope is |0.031| (|0.016| base-corpus-only). The divergence is +0.133.
3. **Observed offsets sign-flip** (+0.018 / −0.054 / +0.133) — no constant fits that shape.
4. **The mechanism is measurably absent.** On the shared held-out English `val.bin`, the
   arith mixture's perplexity penalty **shrinks monotonically and vanishes by 60M**: +2.68
   (p=4e-05) at 11.5M, +1.73 (p=0.22) at 23.2M, **−0.12 (p=0.61)** at 60.3M — where arith
   models are tied with, and numerically ahead of, syntax models. r(penalty, divergence) =
   **−0.74**, opposite the required sign.

Neither prong of the original prediction survives: arith-60M's exposure is not closest to
syntax-23M (it is equidistant from all), and its +0.334 sits **above** the
[+0.201, +0.314] interval rather than inside it.

## Verdict for the write-up

The scale dissociation must be reported as **established in one mixture and not confirmed in
a second**, with the interaction p=0.050 and the underpowered CI both stated. It can no
longer carry "this is the strongest result here" unqualified.

But the failure is **localised, not general**, and that should be said: the mixtures *agree*
at 11.5M (+0.018, p=0.80) and 23.2M (−0.054, p=0.10); the entire disagreement is the single
60.3M cell (+0.133, p=0.098, n=3). With the exposure confound excluded, the two live
candidates are sampling noise at n=3 (realized power 0.21) and a genuine mixture × scale
interaction (p=0.050) — and no further re-analysis can separate them.

**Revised design for the replication** (`scripts/agreement_60M_power.py`, realized SDs):
seeds at 60.3M dominate seeds spread across sizes.

| design | new runs | power |
|:--|--:|--:|
| +3 seeds at 60.3M (n=6) | 3 | **0.88** on +0.133 / 0.66 on a conservative +0.100 |
| +5 seeds at 60.3M (n=8) | 5 | **0.97** / 0.82 |
| +1 seed at every size (n=4), original plan | 3 | 0.57 on the interaction |
