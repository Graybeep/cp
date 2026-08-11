# Week 9 — computed statistical backbone (arith + circuit clusters)

Owner: `scripts/week9_stats_backbone.py`. Re-runnable, pure stats over recorded per-seed endpoints.


## C1 Detect — per-seed paired deficit (primary)

| size | early anchor | delayed null | deficit mean±SD | paired t (df2) | p | Holm | g |
|:--|:--|:--|:--|--:|--:|--:|--:|
| 11M | 0.538 ± 0.018 | 0.357 ± 0.070 | 0.181 ± 0.087 | 3.60 | 0.0345 | — | 1.19 |
| 23M | 0.791 ± 0.007 | 0.577 ± 0.051 | 0.214 ± 0.055 | 6.74 | 0.0107 | 0.0107 | 2.22 |
| 60M | 0.829 ± 0.027 | 0.600 ± 0.040 | 0.229 ± 0.037 | 10.64 | 0.0044 | 0.0087 | 3.51 |

Size moderation (60M vs 23M deficit): p = 0.72, **MDE at 80% power = 0.144** — i.e. only size effects larger than 0.14 are excluded, against a deficit of ~0.21.


## C2 Reopen — endpoints, never collapsed into a ratio

| size | treated (lr_shift) | delayed null | early anchor | gate 0.90×d0 | margin | pass | lift p |
|:--|:--|:--|:--|--:|--:|:--:|--:|
| 11M | 0.747 ± 0.021 | 0.357 ± 0.070 | 0.538 ± 0.018 | 0.484 | +0.262 | PASS | 0.0052 |
| 23M | 0.767 ± 0.014 | 0.577 ± 0.051 | 0.791 ± 0.007 | 0.712 | +0.055 | PASS | 0.0065 |
| 60M | 0.796 ± 0.022 | 0.600 ± 0.040 | 0.829 ± 0.027 | 0.746 | +0.050 | PASS | 0.0027 |

### n=1 arms vs the null's cross-seed range [0.524, 0.626]

| arm | endpoint | verdict |
|:--|--:|:--|
| reset_embedding | 0.586 | no detectable recovery (n=1) |
| reset_plus_spike | 0.576 | no detectable recovery (n=1) |
| lr_spike | 0.536 | no detectable recovery (n=1) |
| cram | 0.574 | no detectable recovery (n=1) |
| freeze | 0.332 | harmful (below entire null range, n=1) |

60M forgetting: breach +1.670 ± 0.104 ppl (gate +1.0, tripped all seeds) vs matched-LR lead +2.477 ± 0.240 → **BOUNDED-NOT-EXCLUDED**, residual bounded ≤ 1.67 ppl.


## C4 Circuit identity — deficit depth vs ablation transfer

Spearman ρ = -0.714, **exact permutation p = 0.0681** (n = 6; asymptotic p = 0.1108).

| dropped point | ρ | exact p | n |
|:--|--:|--:|--:|
| 23M-s2 | -0.700 | 0.1167 | 5 |
| 60M-s2 | -0.900 | 0.0417 | 5 |
| 23M-s3 | -0.900 | 0.0417 | 5 |
| 60M-s1 | -0.700 | 0.1167 | 5 |
| 60M-s3 | -0.500 | 0.2250 | 5 |
| 23M-s1 | -0.500 | 0.2250 | 5 |

Leave-one-out robust at α=0.05: **False** — 4 of 6 single-point deletions push it above 0.05, so it is carried by the whole six-point set rather than being robust within it.


### Contamination correction (2026-07-28)

| triad | deficit superseded → clean | transfer superseded → clean |
|:--|:--|:--|
| 23M-s2 | 0.182 → 0.182 | 0.71 → 0.66 |
| 60M-s2 | 0.250 → 0.202 | 0.40 → 0.40 |
| 23M-s3 | 0.206 → 0.210 | 0.87 → 0.87 |
| 60M-s1 | 0.224 → 0.224 | 0.45 → 0.45 |
| 60M-s3 | 0.272 → 0.260 | 0.31 → 0.20 |
| 23M-s1 | 0.292 → 0.292 | 0.19 → 0.19 |

Superseded pooled Spearman was ρ = -0.943 (exact p = 0.0083); seed-1 triads are unchanged at both sizes (no-op control).


### Cross-harness agreement (same checkpoints, two eval harnesses)

- corrected draw, all 6 triads: mean |Δ| 0.0040, max 0.0080 (n=18)
- superseded, contaminated triads only: mean |Δ| 0.0252, max 0.0620 (n=12)
- binomial prediction for two independent 500-item draws: SD 0.0310

**Measurement floor is ~0.005 on identical items** — not 0.03; that spread was the contaminated draw, not harness disagreement.

By size: 23M 0.571 ± 0.349 vs 60M 0.350 ± 0.131, Welch p = 0.39; SD ratio 2.7×.

**Underpowered, not null:** MDE at 80% power = 0.809, vs an outcome span of 0.681 across all six triads. The design could not have detected any realizable difference, so p = 0.39 excludes nothing. Estimable (deficit no longer collinear with size) ≠ answered.
