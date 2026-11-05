# Pre-registration — five intervention arms to n = 3

**Written 2026-08-11, before any seed-2/3 run of these arms existed.**
Batch: `scripts/intervention_arms_seeds23_batch.sh`.

## 1. What is being closed

Five Tier-1 arms are n = 1 in the Week 4 pilot:

| arm | seed-1 endpoint | current reading | current label |
|:--|--:|:--|:--|
| reset (embedding) | 0.586 | no detectable recovery | SUGGESTIVE (n=1) |
| reset + spike | 0.576 | no detectable recovery | SUGGESTIVE (n=1) |
| LR spike | 0.536 | no detectable recovery | SUGGESTIVE (n=1) |
| cram | 0.574 | no detectable recovery | SUGGESTIVE (n=1) |
| **freeze** | **0.332** | **harmful** | SUGGESTIVE (n=1) |

`freeze` is the one carrying a **directional** claim, and it is the reason this
is Tier 1 rather than a nicety: "harmful" asserts an effect, and an asserted
effect on one run is the single most flaggable thing in §4.4.

`lr_shift` (0.767 ± 0.014) and `cram_lrshift` (0.781 ± 0.031) are already n = 3
and are **not** re-run. Nothing about the mechanism claim is re-opened here.

## 2. Design

Seeds 2 and 3 of each arm, forked from `arith_delay50_exp20_size23M_seed{S}` at
step 2500 — the same common checkpoint each seed's own null and `lr_shift` arm
used — with the pilot's pre-registered defaults unchanged (spike peak 6e-4 over
300 steps; freeze and cram windows 500 steps; reset target `embedding`).
10 forks, ~13 min each.

Comparators, already in hand at n = 3:

| | endpoint |
|:--|:--|
| delayed null | 0.577 ± 0.051 (per-seed range [0.524, 0.626]) |
| early anchor | 0.791 ± 0.007 |
| reopening gate (0.90 × anchor) | **0.712** |

## 3. Decision rules — fixed here, applied once all 10 runs land

Per the standing reporting rule, every arm is reported as **absolute treated
endpoint ± SD beside the null endpoint ± SD**, two numbers, never collapsed into
a ratio and never as percent-of-gap (the denominator is the seed-variable null).

**Reopening** (the pre-registered gate, unchanged): arm mean ≥ 0.712.

**"No detectable recovery"** — promotable from SUGGESTIVE to ESTABLISHED only if
all three hold:
1. arm mean < 0.712 (fails the gate), and
2. the per-seed paired lift against that seed's own null straddles zero, or is
   uniformly negative, and
3. direction is consistent across all three seeds.

If the sign disagrees across seeds, the arm stays SUGGESTIVE and is written as
"no consistent direction at n = 3", not as a failure.

**"Harmful" (freeze)** — promotable only if the endpoint is below the null at
**every** seed, paired per-seed differences are negative at **all three**, and
the arm mean sits below the null's cross-seed range [0.524, 0.626]. Anything
less and the word "harmful" comes out of the paper and is replaced by the
observed endpoints.

**Forgetting check.** Every arm reports held-out perplexity alongside accuracy at
n = 3, under the same rule the reopening arms use: a recovery bought with
general-capability loss is forgetting, not reopening. This matters most for
`reset` and `reset_spike`, which the pre-registration predicted would carry the
biggest forgetting risk.

## 4. Multiplicity

Five arms × one endpoint comparison each = a family of five, Holm-corrected
within it. This family is **separate** from the Detect and Reopen families
already declared in the stats backbone; it is registered here as its own.
The n = 3 arms (`lr_shift`, `cram_lrshift`) are not members — they were tested
under the previously declared Reopen family and are not re-tested.

## 5. What this cannot do

n = 3 does not make these arms well powered against a *small* recovery. The
achieved design detects roughly what the reopening gate encodes (a ~0.14 lift
off the null); a genuine +0.03 recovery would be missed. A null arm is therefore
written as "does not reach the reopening gate", never as "has no effect".
