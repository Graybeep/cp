# H2/H3 discriminator at 23M (confirmatory size)

Is `lr_shift` a generic optimizer booster, or is its benefit specific to
the delayed model's exposure profile? Same intervention, applied to the
EARLY (d0) run, forked from the identical step-2500 checkpoints.

## Endpoints (step 5000, arith exact-match), mean ± SD over seeds 1-3

| condition | endpoint | per-seed |
|:--|--:|:--|
| d0 (early anchor) | 0.791 ± 0.007 | [0.798, 0.792, 0.784] |
| d50 (delayed null) | 0.577 ± 0.051 | [0.524, 0.626, 0.582] |
| d50 + lr_shift (reopened) | 0.767 ± 0.014 | [0.752, 0.778, 0.772] |
| **d0 + lr_shift (DISCRIMINATOR)** | **0.783 ± 0.025** | [0.758, 0.808, 0.784] |

## Decomposition (each lift over its OWN null)

| component | mean | per-seed |
|:--|--:|:--|
| delayed lift = lr_shift(d50) − d50 | +0.190 | [0.228, 0.152, 0.19] |
| generic H2 = lr_shift(d0) − d0 | -0.008 | [-0.04, 0.016, 0.0] |
| alignment H3 = delayed − generic | +0.198 | [0.268, 0.136, 0.19] |

H3 share: 104%   H2 share: -4%
Alignment paired t(2) = 5.168, one-sided p = 0.0177, signs [1, 1, 1]

11M precedent (exploratory): delayed +0.389 = generic +0.115 + alignment +0.274 (70% H3)

Sensitivity — pairing seed 1 against the as-logged null (0.506, the model lr_shift was forked from) instead of the canonical 0.524: delayed +0.196, alignment +0.204 (104% H3). The verdict does not depend on the choice.

## Verdict

H3 DOMINANT (exposure-alignment). lr_shift lifts the DELAYED run +0.190 but the EARLY run only -0.008, so 104% of the reopening effect requires the delayed-exposure profile; only ~-4% is a generic optimizer boost.

## Held-out perplexity (does the discriminator arm pay a cost?)

d0 48.550 ± 1.749 → d0+lr_shift 46.443 ± 0.845
