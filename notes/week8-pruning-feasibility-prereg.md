# Pre-registration — pruning-based localization: FEASIBILITY PILOT (not the full build)

**Written 2026-07-27, before running anything.** Fixed before any pruning number was seen.

## Why a feasibility pilot at all

CLAUDE.md lists pruning-based localization as Week-8 tooling. But the
CROSS-METHOD standing rule has already been paid for twice on *this exact contrast*
(23M arithmetic triads, seeds 1–3): **CKA** stayed 0.66–0.82 regardless of verdict, and
**single-head activation patching** was flat (ratio 0.59–0.79, SD 0.083) with r = 0.045
against the primary readout. Both failed for the same underlying reason — near-floor or
near-ceiling dynamic range, so the metric divided two numbers with no resolving power.

Pruning could easily be a third method hitting the same floor. This pilot asks only:
**does pruning discriminate the known-reorganized case (seed 1, ablation transfer 0.187)
from the known-repair cases (seeds 2 and 3, 0.710 / 0.868)?** If not, we do not build it.

## Method under test

Distinct from mean-ablation by construction: ablation ranks components **one at a time and
marginally**; pruning discovers a **jointly minimal sufficient subnetwork**. Learned gates
g ∈ [0,1], one per attention head and per MLP block (6×8 + 6 = 54 gates at 23M), model weights
frozen; gate g interpolates the component toward its own mean activation (g = 0 is exactly
mean-ablation, so the two methods share a baseline convention and differ only in
marginal-vs-joint selection). Objective = arithmetic answer-span CE + λ‖g‖₁. Binarize at 0.5
→ mask. Primary cross-model metric = **IoU (Jaccard) of mask(A) and mask(C)**.

## Gates — ALL FOUR must pass for GO

1. **Reproducibility floor (the check CKA and patching never had).**
   Fit the mask on A **twice**, independent gate init + data order → IoU(A, A′) ≥ **0.70**.
   If the metric cannot reproduce itself on identical weights, no cross-model IoU means
   anything. This gate runs first; failing it stops the pilot.
2. **Dynamic range.** Across the three seeds, max − min of IoU(A,C) ≥ **0.20**, and the three
   values must not sit inside a single 0.10-wide band. (Ablation transfer's own range on these
   triads is 0.187–0.868.) Compression near 0 or 1 = FAIL, exactly as for CKA.
3. **Ordering vs the primary readout.** Seed 1 must have the **lowest** IoU(A,C) of the three.
   **n = 3 cannot establish a correlation** — no p-value is claimed or accepted here; ordering
   plus range-vs-floor is the whole feasibility read. Anything stronger needs the full build.
4. **Functional validity of the mask.** At the selected sparsity, masked-A retains ≥ **80%** of
   A's exact-match accuracy, and the **complement** mask drops A to ≤ 0.15. Otherwise the
   "circuit" is not a circuit and gates 1–3 describe an artifact.

## Decision rule

- **All four pass → GO**: build the full Week-8 pruning localization (all 6 triads, both sizes).
- **Any fail → NO-GO**: record pruning as a **third non-discriminating method on this contrast**,
  do not build it, and keep the standing-rule position that mean-ablation is the *only* probe
  with resolving power here — reported explicitly as a single-method finding.

A NO-GO is a real result, not a wasted pilot: it is the third confirmation that this contrast
is hard to resolve, and it is cheap (eval-only, no training, checkpoints already on disk).

## Pre-committed interpretation of the awkward case

If gate 1 passes but gates 2–3 fail *while masks look sensible* (gate 4 passes), the reading is
**"pruning measures something real but not circuit identity"** — i.e. the subnetwork is
recoverable and stable, yet A and C overlap the same amount whether or not C rebuilt. That is a
NO-GO for the diagnosis question, and it is NOT to be reported as evidence of repair.
