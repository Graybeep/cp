# Pre-registration — Syntax LD Model-Size Sweep (reposition hypothesis)

Status: PRE-REGISTERED DESIGN (2026-07-19), fixed BEFORE any sweep number
exists. Same discipline as every other pre-registered call in this project
([[syntax-exp20-preregistered-read]], Week-3 pilot reads): the criteria for
BOTH confirming and disconfirming reposition are committed here, in advance,
so the read can't be rationalized after the fact — in EITHER direction. This
document exists specifically because the project just came within one
unverified exclusion of dismissing a real signal; the mirror risk (treating a
lone high seed at a larger size as confirmation) is pre-banned below.

## The hypothesis, stated as a falsifiable rate

Observed at 23M (delay-0, continuation-scored LD, gap_region_interaction.py):
8 seeds → 7 clustered 0.520–0.544 (mean 0.534, SD 0.009) + 1 reproducible high
draw at ~0.583 (seed 1, verified 0.586→0.583 across two trainings). So the
23M high-acquisition RATE is **P(high) ≈ 1/8 = 12.5%**.

- **Reposition (H1):** the high-acquisition mode is SCALE-DEPENDENT — P(high)
  increases with model size (and/or the whole LD distribution shifts up), i.e.
  the mode becomes the typical outcome as the model grows.
- **Null (H0, the DEFAULT):** P(high) is size-INDEPENDENT — a rare draw that
  occurs at roughly the same ~12% rate at every scale (some seeds get lucky
  regardless of size). Under H0, syntax LD stays deprioritized.

H0 is the default. H1 must EARN rejection of it. A result that fails to reject
H0 is a disconfirmation of reposition, not "inconclusive, lean reposition."

## Unit of evidence: the RATE across seeds, never a single seed

**PRE-BANNED as confirmation:** one (or a few) high seed(s) appearing at a
larger size, absent a rate increase. 1/12 high at 60M is the SAME rate as 1/8
at 23M ≈ H0. This is the exact mirror of the seed-1 exclusion mistake and does
not count as evidence for reposition. Confirmation requires a change in the
RATE or a shift of the DISTRIBUTION — quantities that only exist across many
seeds per size.

## Design (delay-0 only; this is an acquisition-RATE sub-study)

Delay is irrelevant here — the question is whether LD is learnable-enough to
have a mode that scales, not whether it has a critical period. Delay×size for
syntax is only funded AFTER this sub-study rejects H0 (gate below).

- **Model sizes — RESOLVED: option B (extended), user-approved 2026-07-19
  after a measured VRAM check (verify-don't-infer).** Three log-spaced sizes:
  - **~11.5M** — 4 layers / 256 dim / 8 heads (in-spec).
  - **~23.2M** — 6 / 384 / 8 (the existing pilot config; runs already exist).
  - **~60.3M** — 8 / 640 / 10 (extends beyond the locked 30M ceiling for THIS
    sub-study only; deliberate, to give the rate a chance to move).
  VRAM CONFIRMED (not inferred): a real bf16 forward+backward+AdamW step at
  micro-batch 8 × seq 512 for the 60.3M config peaks at 2.95 GB allocated /
  3.06 GB reserved on the 8.59 GB RTX 4060 → 5.5 GB headroom (23M anchor:
  1.71/2.00 GB, matches its known footprint, so the probe is representative).
  Probe: `scripts/vram_probe.py`. The 30M-ceiling deviation is scoped to this
  acquisition-rate sub-study; the main Detect grid's size axis is unaffected
  unless this sweep confirms reposition.
- **Seeds per size: 24 (DERIVED from the simulation power analysis below, not
  a round number).** 24×3 = 72 delay-0 runs ≈ 23 min each ≈ 28 h, batchable
  over ~3 nights. See the power section for why 24 and not 12.
- **Everything else identical** across sizes (exp10, pool v1, 5000 steps, same
  vocab.bin, same eval): only width/depth changes, per the size-tag convention.
- **Metric:** continuation-scored LD (gap_region_interaction.py), final ckpt
  (step 5000), same as the 23M seeds.

## Operational "high-acquisition seed" — threshold DERIVED, not hand-placed

The binary "high" cutoff is the most load-bearing number here: P(high), whose
trend the whole decision rests on, only means something if the cutoff cleanly
separates two real populations rather than being a line drawn near the one
existing high value. So it is DERIVED by a rule and stress-tested, not eyeballed.

**Threshold = μ_low + 3·σ_low ≈ 0.534 + 3(0.009) = 0.561 (→ use 0.56).** This is
the ~99.7th percentile of the LOW (null) cluster — an exceedance threshold
referenced to the NULL distribution (the 7-seed low mode), NOT to seed 1's high
value. A seed above it is >3σ beyond the null band. μ_low, σ_low are re-estimated
from the SMALLEST model's seeds (11.5M, the cleanest presumed-all-low null) once
the sweep runs; if that recomputation moves the cutoff, the moved value is used
(it is a rule, not the literal 0.56).
- **Validated as the min-misclassification boundary (sim, power_sim.py):** at
  0.56 the classified rate tracks the true rate (true 0.12→0.12, 0.40→0.39,
  0.70→0.68); at 0.55 it inflates (low-mode leakage: 0.12→0.16); at 0.58 it
  deflates (clips the high mode centred 0.583: 0.40→0.24). 0.56 sits in the
  valley between N(0.534,0.009) and N(0.583,0.012).
- **Robustness requirement (guards against a knife-edge artifact):** the CONFIRM
  call must hold at ALL of thr ∈ {0.55, 0.56, 0.57}. The rate ordering across
  sizes is preserved over this band, so a real trend survives it; a line-placement
  artifact would not.
- **CAVEAT (honest):** the min-misclassification optimality assumes two real
  populations, but the high mode is presently a ONE-seed characterization (seed 1,
  reproduced — location solid, spread assumed). Adding 24 seeds/size at 11.5M and
  23M directly tests whether that second population is real; if the high mode does
  not reappear with more seeds, bimodality is not established and the P(high)
  framing is reconsidered (fall back to reporting the raw LD distribution/trend).
- **Report the continuous LD distribution and the CA trend at all three
  thresholds** so no single line is solely load-bearing.
- **Validity guard:** a seed counts as "high" only if the interaction is a
  genuine gap contrast — BOTH with_gap and no_gap continuations contribute
  (as in seed 1: 0.306 / 0.859), not one paradigm collapsing to a prior.
- **Reproduce-the-top-draw rule:** before the single highest seed at the top
  size is used in any headline claim, re-run it once (as done for seed 1).

## Power analysis (simulation, `scripts/power_sim.py`, 4000 runs/cell)

Confirm rule = CA one-sided trend p<0.05 AND top-size observed rate ≥ 0.40.
Rates per size given as (11.5M, 23M, 60M). Type-I (H0 flat 0.12/0.12/0.12) is
0.00–0.05 for the confirm rule at every N — well controlled.

| scenario                         | N=12 | N=18 | N=24 | N=30 | N=40 |
|----------------------------------|------|------|------|------|------|
| STRONG   .12/.40/.70 (confirm)   | 0.93 | 0.98 | 1.00 | 1.00 | 1.00 |
| GRADED   .12/.26/.40 (CA trend)  | 0.34 | 0.48 | 0.61 | 0.69 | 0.82 |
| MARGINAL .12/.12/.40 (confirm)   | 0.40 | 0.37 | 0.45 | 0.51 | 0.51 |

Reading:
- The gate is **well-powered (≥0.93 at N≥12) for a STRONG reposition** — the
  mode becoming the majority, which is the fundable case. Even N=12 suffices here.
- It is **structurally unable to confirm a MARGINAL effect** sitting at the 0.40
  floor (≈0.5 at ALL N — the observed rate is centred on the threshold, so it
  clears it ~half the time; N=40 doesn't help). This is an ACCEPTED conservative
  bias: a gate protecting expensive grid budget should not greenlight a marginal
  minority-mode effect. A disconfirm under a true-marginal effect is a deliberate,
  documented false-negative, not an oversight.
- **N=24 is chosen** because more seeds buy what the marginal case can't: graded-
  trend sensitivity (0.34→0.61) and a usable reporting CI — Wilson 95% for an
  observed 0.40 shrinks from [0.19,0.68] at N=12 (overlaps 25%) to [0.24,0.61]
  at N=24 (excludes the 12% baseline). Threshold-free Spearman was LESS powerful
  in every cell, so dropping the cutoff is not an option that helps.

## CONFIRM reposition — ALL must hold (pre-committed)

1. **Monotone rate trend:** P(high) non-decreasing across the ordered sizes,
   AND a Cochran–Armitage one-sided trend test has positive slope at
   **p < 0.05** — and this holds at ALL of thr ∈ {0.55, 0.56, 0.57} (threshold-
   robustness, so no single cutoff placement drives the call).
2. **Effect-size floor:** the LARGEST size's P(high) ≥ **0.40** (materially
   above the 23M 12.5% — the mode is becoming typical, not merely nudged). A
   significant-but-tiny trend (e.g. 12%→18%) does NOT confirm; it is reported
   as "weak size sensitivity," which is not reposition. NOTE (see power section):
   this compound rule is powered ≥0.93 for a STRONG effect but ~0.5 for one
   sitting exactly at the 0.40 floor — the gate is deliberately conservative.
Both #1 and #2 are required. Secondary (mechanism descriptor, NON-gating):
mean LD vs size — a rising mean with a fixed tail = "whole distribution
shifts"; a fixed mean with a fattening tail = "mode fraction grows." Report
which, but confirmation rests on the RATE (#1+#2), not the mean.

## DISCONFIRM reposition — any one (the symmetric pre-commitment)

- **Flat rate:** trend test p > 0.05 AND point estimates non-increasing / CIs
  overlapping across sizes → high mode is size-INDEPENDENT → reposition
  DISCONFIRMED → **deprioritize stands**; seed-1-style high draws are rare luck
  at any scale, to be reported as such.
- **Significant but sub-floor:** trend p < 0.05 but top-size P(high) < 0.40 →
  "weak size sensitivity," NOT reposition → deprioritize stands, with the weak
  trend noted as a caveat/future-work item.
- **Non-monotone / noise:** rate wobbles without direction → H0 NOT rejected →
  treated as DISCONFIRMING by default (null wins unless H1 earns it). Optionally
  add seeds ONCE if a pre-stated power calc says the wobble is under-powered;
  otherwise deprioritize.

## Gate on the expensive grid

This delay-0 rate sweep runs and is read BEFORE any syntax delay×size cells are
funded in the Weeks 5–7 grid. CONFIRM → syntax LD earns a delay×size slot and
the size axis is repositioned to characterize the mode's scaling. DISCONFIRM →
syntax LD is dropped from the costly delay×size grid; arithmetic carries Detect.
This mirrors the Week-3 vocab-control exit gate: a cheap measurement decides
whether an expensive phase is worth its budget.

## Knobs marked tunable-before-first-number (then locked, per project rule)

Seeds/size (24, power-derived), high-threshold RULE (μ_low+3σ_low ≈ 0.56, with
the {0.55,0.56,0.57} robustness band), trend-test alpha (0.05), effect floor
(0.40). Size range is RESOLVED (option B, VRAM-confirmed) — not a knob. All
pre-committed here; any change must happen before the sweep's first eval number
is seen, and be recorded with a reason — same standing rule as the rest of the
project. Power basis and threshold-sensitivity: `scripts/power_sim.py`.
