# 60.3M mixture replication — result

**Ran 2026-07-27.** 5 new arithmetic-mixture d0 baselines at 60.3M (seeds 4–8), scored against
the rules fixed in `notes/agreement-60M-mixture-replication-preregistration.md` **before** any
of them was trained. Artifacts: `quartet_scale_arithmix_n8.json`,
`replication_60M_readout.json`, `scripts/agreement_60M_replication_readout.py`.

## Validity gates — both pass

- Re-scoring 60.3M seeds 1–3 reproduces `quartet_scale_arithmix.json` **exactly** (all 9 prior
  models, all three metrics, bit-identical) — the item set and checkpoints did not move.
- Unigram control attraction **+0.0000** exactly.
- Attraction > 0 in all 8 arithmetic 60.3M models.

## PRIMARY TEST → **rule (B): the divergence closes as noise**

Welch two-sided, 60.3M attraction, arithmetic (n = 8) vs syntax (n = 24):

| | mean ± SD | n |
|:--|--:|--:|
| arithmetic mixture, 60.3M | **+0.266 ± 0.092** | 8 |
| syntax mixture, 60.3M | **+0.201 ± 0.056** | 24 |

**diff +0.065, 95% CI [−0.013, +0.143], t = 1.90, df = 8.8, p = 0.091.**

Not significant at α = 0.05 → **verdict (B) as pre-registered.** The gap **halved**, from +0.133
at n = 3 to +0.065 at n = 8. The five new seeds came in well below the original three:

| | mean ± SD |
|:--|--:|
| seeds 1–3 (original) | +0.334 ± 0.082 |
| seeds 4–8 (new) | +0.225 ± 0.077 |

difference −0.108 (p = 0.135). Per-seed: 0.249, 0.341, 0.411 | 0.155, 0.340, 0.246, 0.232, 0.154.
The original three happened to include the two highest draws of the eight.

### Two honest bounds on that null

1. **The CI still contains +0.133.** Upper bound +0.143. So this fails to *confirm* the original
   effect; it does not formally *exclude* it. "Closes as noise" is the pre-registered call and the
   evidence leans strongly that way — but the correct phrase is "does not survive replication,"
   never "is excluded."
2. **Achieved power was slightly under plan.** The realized arithmetic SD came in at **0.092**
   against the **0.0815** used to size the study, so power for a true +0.133 was **0.93**, not
   0.96 (0.81 at +0.110, 0.73 at +0.100). Effects below ≈0.11 remain untested — and note the
   observed +0.065 sits *below* that threshold, so a real-but-small divergence is not ruled out.

Per-size cross-mixture contrasts, none significant: 11.5M +0.018 (p = 0.80), 23.2M −0.054
(p = 0.10), 60.3M +0.065 (p = 0.091). The pooled mixture × log₁₀(params) interaction **weakens
from p = 0.050 to p = 0.140** (+0.109 ± 0.073, n = 86).

## SECONDARY — the verdict splits, and this matters for the write-up

Finding #2 is a *conjunction*: "attraction shrinks with scale **while complexity does not**."
With 60.3M at n = 8 the two halves now have **different** replication status.

| | arithmetic mixture (n = 14) | syntax mixture (n = 72) |
|:--|--:|--:|
| attraction slope | **−0.207/decade, CI [−0.395, −0.019], p = 0.033 OLS / 0.073 robust** | −0.316, p = 2.3e-17 |
| complexity slope | **−0.112/decade, CI [−0.194, −0.029], p = 0.012** | +0.016, p = 0.43 |
| paired (attraction − complexity) slope | **−0.095, CI [−0.314, +0.123], p = 0.36** | **−0.333, p = 3.8e-11** |
| monotone 11 > 23 > 60 | no (0.448 → 0.260 → 0.266) | yes |

- **"Attraction shrinks with scale" now DOES reproduce** in the second mixture — the slope was
  −0.139 (p = 0.28) at n = 3 and is −0.207 (p = 0.033) at n = 14. **Significance is
  estimator-dependent** (added 2026-07-27, `slope_imbalance_check.md`): the −0.21 estimate is
  stable under every estimator, but p = 0.033 only under pooled-variance OLS; a
  heteroscedasticity-robust SE gives **p = 0.073**, and the (3, 3, 8) design's leverage sits on
  the *under*-sampled 11.5M end (0.254/point, SD 0.106), not on the 8-seed cell. Quote as
  "p = 0.033 OLS / 0.073 robust", not as a bare 0.033.
- **The dissociation does NOT.** In the arithmetic mixture complexity falls too, and the paired
  within-model contrast — the correct test, since both components are measured on the same models
  — is flat (p = 0.36). The syntax mixture's −0.333 sits marginally outside the arithmetic CI
  (lower bound −0.314), so this is a *non-replication*, not a refutation: at n = 3 for the two
  smaller sizes it is underpowered, and it does not exclude a weaker real dissociation.
- Monotonicity still fails, now because 23.2M (n = 3) sits slightly *below* 60.3M (n = 8), not
  because 60.3M is high.
- **"Below chance at the smallest size only" survives**: arithmetic 60.3M controlled mismatch is
  **0.626 ± 0.114, above chance (p = 0.017)**.

## What to claim now

1. **The 60.3M divergence does not survive replication at n = 8** (power 0.93 for the originally
   observed +0.133) — **"does not survive replication", never "did not replicate" or "excluded"**:
   the 95% CI [−0.013, +0.143] still contains +0.133, and effects below ≈0.11 remain untested. The
   n = 3 excursion is recorded as sampling variation, and the exposure confound was already
   excluded independently (`exposure_recheck.md`).
2. **The scale trend in attraction reproduces across both mixtures** — directionally and in
   magnitude (−0.21/decade under every estimator), at borderline significance in the second
   mixture (p = 0.033 OLS / 0.073 robust).
3. **The dissociation from complexity remains a single-mixture result**, and is flat under every
   estimator — the robust half of this comparison. It is the *principal* surviving qualification
   on finding #2 and should carry the caveat; the 60.3M cell is **bounded, not closed**
   (unreplicated ≠ excluded, per 1).

Net: the replication cost 5 runs and moved the qualification from "one anomalous cell" to "the
complexity half of the dissociation is unreplicated" — a smaller, better-localised, and more
honest caveat than the one it replaced.
