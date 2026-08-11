# Week-8 circuit cluster — eval-contamination recheck (2026-07-28)

**Trigger.** The Week-9 backbone assembly found that `week8_circuit_identity.py` built its
eval set with `ArithConfig(seed=<run seed>)`. Question raised: was the reported
"seed 3 is deeper yet shows more repair" inversion computed under a mismatched harness?

**Answer: yes, and it was worse than a mismatched draw — it was train contamination.**
`prepare_skills.prepare_arith` holds out exactly one problem set,
`make_eval_set(ArithConfig(seed=1), 1000)` = `data/skills/arith/eval.jsonl`, passed as
`exclude=` when writing `train.bin`. Any other draw overlaps the 200 000 training
problems:

| eval draw | problems seen in training |
|:--|--:|
| seed 1 | **0 / 500 (0.0 %)** — the held-out set |
| seed 2 | 97 / 500 (**19.4 %**) |
| seed 3 | 87 / 500 (**17.4 %**) |

So four of the six triads (23M·2, 23M·3, 60M·2, 60M·3) had every accuracy, every ablation
damage and therefore every transfer ratio measured partly on memorised problems, on an
item set that also differed from triad to triad.

## What was re-run

All six triads re-scored on the held-out draw (`--eval-seed 1`), including **seed 1 at
both sizes as a no-op control** — its draw was already the held-out one, so it must
reproduce exactly. It does, to the digit (23M·1: 0.794 / 0.502 / 0.292 / 0.19;
60M·1: 0.796 / 0.572 / 0.224 / 0.45), which establishes that the eval draw is the only
thing that moved.

| triad | deficit superseded → clean | transfer superseded → clean |
|:--|:--|:--|
| 23M·2 | 0.182 → 0.182 | 0.71 → **0.66** |
| 60M·2 | 0.250 → **0.202** | 0.40 → 0.40 |
| 23M·3 | 0.206 → 0.210 | 0.87 → 0.87 |
| 60M·1 | 0.224 → 0.224 *(control)* | 0.45 → 0.45 |
| 60M·3 | 0.272 → **0.260** | 0.31 → **0.20** |
| 23M·1 | 0.292 → 0.292 *(control)* | 0.19 → 0.19 |

Cross-harness agreement on the corrected draw is **mean |Δ| 0.0040, max 0.0080** across
all 18 model-arm comparisons (vs 0.0252 / 0.062 for the contaminated triads before).
The measurement floor on identical items is therefore **~0.005**, not the ~0.03 the
earlier backbone draft inferred from the contaminated spread.

## 1. The inversion SURVIVES (the question asked)

On the common held-out draw, with bootstrap CIs recomputed on the same items
(`week8_transfer_bootstrap.py --eval-seed 1`):

| 23M triad | deficit | transfer | 95 % CI |
|:--|--:|--:|:--|
| seed 2 | 0.182 (shallower) | **0.659** | [0.611, 0.708] |
| seed 3 | 0.210 (deeper) | **0.868** | [0.832, 0.903] |

CIs remain **disjoint**, and the ordering is unchanged: the deeper-deficit seed shows
*more* repair, not less. The correction slightly **widens** the inversion (transfer gap
0.16 → 0.21; deficit gap 0.024 → 0.028). The finding may go in the paper as stated.

## 2. But the pooled graded relation does NOT survive

This was not asked about and is the more consequential result:

| | superseded | corrected |
|:--|--:|--:|
| pooled Spearman ρ (n = 6) | −0.943 | **−0.714** |
| exact permutation p | 0.0083 | **0.0681** |
| leave-one-out | robust (worst p = 0.042) | **fragile — 4 / 6 deletions give p ≥ 0.05** |

The direction persists but the "deficit depth grades circuit identity" relation is **no
longer significant at n = 6** and is not robust within its own six points.

## 3. The collinearity argument REVERSES (in the paper's favour)

The withdrawn scale claim was withdrawn because "no 23M triad lies in the 60M deficit
window", making size and deficit collinear. On corrected numbers the deficit ranges
**overlap and the means are nearly identical**:

| size | transfer (mean ± SD) | deficit range | deficit mean |
|:--|:--|:--|--:|
| 23M | 0.571 ± 0.349 | 0.182 – 0.292 | 0.228 |
| 60M | 0.350 ± 0.131 | 0.202 – 0.260 | 0.229 |

So size is now **separable** from deficit depth, and the comparison can be made: at
essentially matched mean deficit, transfer **level** does not differ significantly by
size (Welch p = 0.39). This is a better-founded statement than the previous
"unidentifiable" — the question is now answerable and the answer is null on level.

> **CORRECTION 2026-08-03 — the last sentence above is wrong and is superseded.**
> The correction established **estimability**, not **power**, and the two were
> conflated. Running the same `mde_two_sample` helper used for C1: pooled SD 0.263,
> **MDE at 80 % power = 0.809**, against a transfer ratio whose six triads span only
> 0.187–0.868 (span 0.681); Welch 95 % CI on the difference [−0.537, +0.979].
> The design could not have detected *any* difference realizable within the measured
> range, so p = 0.39 excludes nothing. Compare C1, where MDE 0.144 < the 0.21 deficit
> and the null genuinely constrains.
> **Read this section as: the question became estimable and remains unanswered.**
> "Transfer level does not differ by size" is now itself a banned phrasing
> (`week9-stats-backbone.md` §7); the MDE must travel with the p.

What remains directional is **reproducibility** (SD 0.349 vs 0.131, ratio 2.7×, down from
5.1×) — at n = 3 per group a variance comparison is not inferentially useful, so this
stays SUGGESTIVE.

The 23M-only extrapolation check no longer supports anything: the 23M fit is
r = −0.85 (p = 0.351) and its residuals against the 60M points are −0.30 / −0.15 / −0.20,
all one-sided. Drop it as evidence.

## 4. Tooling fixed so this cannot recur

- `week8_circuit_identity.py`: `--eval-seed` now defaults to **1** (the held-out set)
  regardless of run seed, plus a **fail-loud guard** that refuses to run when any eval
  problem is absent from `eval.jsonl`, overridable only with
  `--allow-contaminated-eval` (needed to reproduce a pre-fix run). Contamination
  fraction is recorded in every report.
- `week8_transfer_bootstrap.py`, `week8_scale_vs_deficit.py`: `--eval-seed` /
  `--tag-suffix` so corrected and superseded artifacts coexist rather than overwrite.
- **Same pattern still present, not re-run:** `week8_causal_trace.py:142` and
  `week8_pruning_feasibility.py:191` build eval sets the same way. Neither supports a
  surviving claim — patching was retracted as non-discriminating and pruning was a
  feasibility NO-GO — so no conclusion depends on them. Both must be re-run on the
  held-out draw before any future use.

Artifacts: `seed{1,2,3}[_60M]_cleaneval.json`, `scale_vs_deficit_cleaneval.json`,
`transfer_bootstrap_23M_cleaneval.json`, `logs/week9_stats/backbone_stats.{json,md}`,
`logs/week9_stats/cleaneval_recheck.json`.
