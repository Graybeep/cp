# Pre-registration — decomposing `lr_shift` into level and shape

**Written 2026-08-11, before any decomposition arm was run.**
Batch: `scripts/lr_decomposition_batch.sh`. Instrument:
`reopen.interventions.LRComponentWindow`.

## 1. What is being closed

The mechanism claim — closure at d50 is "largely a property of the optimizer's
decayed learning-rate schedule" — rests on a single lever. `ShiftedLRWindow`
replays the early schedule segment, which simultaneously changes:

1. the **level** (window-mean LR rises from 1.692e-4 to 4.873e-4, ×2.88), and
2. the **shape** (a warmup ramp from ~0 plus first-half cosine curvature,
   instead of continued monotone decay).

"Which of those is doing the work" is unanswered, and it is the difference
between a black-box result and a mechanistic one.

## 2. Design — a 2×2, not two ad-hoc ablations

| | level = late (1.692e-4) | level = early (4.873e-4) |
|:--|:--|:--|
| **shape = flat** | **C** `lr_flat_control` | **A** `lr_level_only` |
| **shape = early segment** | **B** `lr_shape_only` | `lr_shift` — *already n=3* |

Level is matched on the **window mean**, not the peak: the mean is the only
level statistic a flat arm can express, and a peak-matched flat arm would carry
a different area under the schedule and stop being a level control.

The fourth cell is the existing `lr_shift` arm, and the implementation is
asserted in `scripts/test_interventions.py` to reproduce `ShiftedLRWindow`
**exactly** (normalisation collapses to scale 1.0). The decomposition is
therefore anchored to the arm it explains rather than being a re-implementation
that might drift from it.

**C is load-bearing.** Without a flat arm at the untreated level, A confounds
"higher LR" with "no longer decaying". A−C isolates level; B−C isolates shape.

3 arms × 3 seeds = 9 forks from the same common checkpoints
(`arith_delay50_exp20_size23M_seed{1,2,3}` @ step 2500), ~13 min each.

## 3. Comparators (in hand, n = 3)

| | endpoint |
|:--|:--|
| delayed null (untreated) | 0.577 ± 0.051 |
| `lr_shift` (both components) | 0.767 ± 0.014 |
| early anchor | 0.791 ± 0.007 |
| reopening gate (0.90 × anchor) | 0.712 |

## 4. Decision rule — fixed before the data

All arms reported as absolute endpoint ± SD beside the null's, never as
percent-of-gap. Per-seed paired lifts against each seed's own null.

Let A, B, C be arm means and L = 0.767 (`lr_shift`).

- **Level-driven** if A ≥ 0.712 (clears the gate) **and** B < 0.712, and A − C
  exceeds B − C at every seed.
- **Shape-driven** if B ≥ 0.712 and A < 0.712, with the per-seed ordering
  reversed.
- **Both required (interaction)** if neither A nor B clears the gate while
  `lr_shift` does — i.e. the components are jointly but not separately
  sufficient. This is a positive finding, not a null, and is the outcome most
  consistent with the warmup ramp mattering only because it is followed by a
  high-level decay.
- **Additive** if A − C and B − C sum to within one null SD (0.051) of L − C.
  Report the decomposition as approximately additive and give both components.

If C itself clears the gate, the whole framing changes: flatness alone at the
*untreated* level would mean the deficit is about continued decay per se and
neither component is needed. That outcome is pre-registered as
**"decay-arrest, not level or shape"** and would supersede §4.4's wording.

**Multiplicity.** Three arms, one endpoint comparison each, Holm-corrected
within this family of three. Declared separately from the Detect, Reopen, and
n=3-arms families.

**Forgetting check.** Held-out perplexity reported for every arm, as everywhere
else: an arm that clears the accuracy gate while degrading perplexity beyond the
matched threshold is scored as forgetting, not reopening. Arm A runs at ~2.9×
the untreated LR for 2500 steps and is the one to watch.

## 5. What this cannot do

n = 3 per cell resolves effects on the scale of the reopening gate (~0.14 off
the null), not small ones. A component contributing, say, +0.03 will be
reported as "not separately sufficient", which is a statement about the gate,
not evidence that the component is inert. The 2×2 also holds the *onset*
(step 2500) and the *window* (to end of training) fixed; nothing here speaks to
other onsets.
