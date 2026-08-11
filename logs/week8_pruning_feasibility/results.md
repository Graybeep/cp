# Pruning-based localization — FEASIBILITY PILOT: **NO-GO**

Ran 2026-07-27, eval-only (no training), 23M triads seeds 1–3.
Gates pre-registered in `notes/week8-pruning-feasibility-prereg.md` before any number was seen.
Script: `scripts/week8_pruning_feasibility.py`. Raw: `results.json`, `results_lam0.01.json`,
`results_lam0.06.json`.

**Question:** does pruning separate the known-**reorganized** triad (seed 1, ablation transfer
0.187) from the known-**repair** triads (seeds 2/3, 0.710 / 0.868)? Asked because CKA and
single-head patching both already failed on this exact contrast.

**Method:** learned gates over 54 components (6 layers × 8 heads + 6 MLPs), model weights frozen,
gate interpolating each component toward its own mean activation (g = 0 ≡ mean-ablation, so the
intervention matches the primary method and only marginal-vs-joint *selection* differs).
Objective = answer-span CE + λ‖g‖₁, binarize at 0.5, compare mask(A) vs mask(C) by IoU.

## Result at the pre-registered λ = 0.03

| seed | ablation transfer (primary) | IoU(A,C) | IoU(A,A′) floor | mask retains A's acc | complement |
|--:|--:|--:|--:|--:|--:|
| 1 (reorganized) | 0.187 | 0.442 | 1.000 | 0.80 | 0.000 |
| 2 (repair) | 0.710 | 0.513 | 1.000 | 0.94 | 0.000 |
| 3 (repair) | 0.868 | 0.425 | 1.000 | 0.81 | 0.000 |

| gate | result | |
|:--|:--|:--|
| 1. reproducibility floor ≥ 0.70 | **PASS** | IoU(A,A′) = **1.000** — two independent fits give the *identical* mask |
| 2. dynamic range ≥ 0.20 | **FAIL** | spread 0.088, all three inside one 0.10-wide band |
| 3. seed 1 lowest | **FAIL** | seed 3 is lowest (0.425); seed 1 sits in the middle |
| 4. functional validity | **PASS** | mask keeps 80–94% of accuracy on ~30/54 components; complement → 0.000 |

**VERDICT: NO-GO.** This is the pre-committed "awkward case": the metric is *real* — perfectly
reproducible and functionally valid, the mask genuinely is a sufficient arithmetic subnetwork —
but it does not track circuit identity. A and C overlap about the same amount whether or not C
rebuilt.

## The sparsity sweep, and why it makes the NO-GO stronger

Run to pre-empt "you picked the wrong λ". It found something worse than insensitivity:

| λ | IoU seed 1 / 2 / 3 | spread | **r vs ablation transfer** | mask retains A's acc | gate 4 |
|--:|:--|--:|--:|:--|:--|
| 0.01 | 0.609 / 0.511 / 0.523 | 0.098 | **−0.944** | 0.89 / 0.99 / 0.94 | pass |
| 0.03 | 0.442 / 0.513 / 0.425 | 0.088 | **+0.118** | 0.80 / 0.94 / 0.81 | pass |
| 0.06 | 0.333 / 0.529 / 0.472 | 0.196 | **+0.872** | 0.67 / 0.70 / 0.83 | **FAIL** |

Seed 1's IoU moves monotonically from **highest** (0.609) to **middle** to **lowest** (0.333) as
the mask is tightened. **The sign of the method's answer is set by a hyperparameter**, not by the
models. At λ = 0.01 pruning "says" seed 1 reused A's circuit *more* than the repair cases; at
λ = 0.06 it "says" the opposite.

And the λ that agrees with the primary readout (r = +0.872) is exactly the λ where the mask
**stops being a valid circuit** — retention falls to 0.67 / 0.70, below the pre-registered 0.80
bar. At that sparsity the masks are over-pruned, so IoU differences plausibly track *how much
accuracy each mask sacrificed* rather than which components each model uses.

**This is the trap the standing rule exists for.** Had this pilot been run at λ = 0.06 only, it
would have reported "seed 1 lowest overlap, r = +0.87 with ablation — pruning independently
confirms reorganization" — the same shape of false corroboration as single-head patching at
seed 1, and it would have been wrong for the same reason (unvalidated across the contrast set).

## Consequence

- **Do not build the full Week-8 pruning localization.** Three methods (CKA, single-head
  patching, pruning) now fail to discriminate on this contrast.
- Pruning fails *differently*, and the difference is worth one sentence in the paper: CKA and
  patching were **compressed near a floor/ceiling**; pruning is stable and functionally valid but
  **hyperparameter-determined in sign**. A method can be reproducible, causally validated, and
  still have no resolving power for the question asked.
- The standing-rule position is unchanged and now better supported: **mean-ablation transfer is
  the only probe with resolving power here, and the circuit-identity finding is single-method.**

## Limits of this pilot

n = 3 triads at 23M, one architecture, head/MLP granularity, one binarization threshold (0.5).
No correlation is claimed from n = 3 — the r values above are descriptive, reported only to show
that the *sign* moves with λ. A NO-GO here does not mean pruning is useless in general; it means
it does not resolve *this* contrast at *this* scale, which is all the feasibility question asked.
