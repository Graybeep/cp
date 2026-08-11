# Week 8 circuit-diagnosis — seed-scope pre-registration

**Decided 2026-07-21, before any diagnosis tooling is built.** Locks how many
seeds the circuit-identity analysis covers, so the activation-patching / CKA /
ablation / pruning tooling is not built around seed 1 by default.

## The problem this resolves

Seed 1 is atypical in TWO ways simultaneously:
1. **Deepest-hit deficit** — 0.274 vs 0.166 (seed 2) / 0.202 (seed 3).
2. **The fork source** — every Week-4 intervention resumed from
   `arith_delay50_exp20_size23M_seed1@2500` (common-checkpoint rule).

A circuit-identity verdict ("repaired/re-plasticized" vs "new circuit") measured
ONLY on seed 1 entangles the finding with "deepest-damaged case" — it cannot
distinguish an intervention effect from a severity effect. This is the same
single-seed generalization trap Detect and Reopen both fell into (seed variance
dominated both); it must not be repeated at the Diagnose layer.

## Decision

**Build seed-parametric; run FULL battery on seed 1; run a PARTIAL confirmatory
pass on seed 2; do NOT scope the headline circuit-identity claim to seed 1.**

### Scope split
| Diagnostic | Seeds | Why |
|---|---|---|
| Activation patching (baseline→recovered) + ablation transfer | **1 & 2** | This IS the "repaired vs new circuit" verdict — the Diagnose headline; must not rest on n=1. |
| CKA layer sweep, pruning-localization, per-head fine mapping | **1 only** | Localizing/characterizing tooling; informative at n=1 for a pilot; where build+run cost concentrates. |

### Why seed 2 (not seed 3)
Seed 2 is the SHALLOWEST deficit (0.166) → maximal contrast to seed 1's 0.274 →
most power to expose a severity-moderated circuit-identity effect with one added
seed. Seed 3 (0.202) is intermediate = weaker contrast.

### Why it's cheap (no retraining)
Seed-2's full triad already exists on disk (all step-5000, 266MB each):
- early baseline (delay-0): `checkpoints/arith_delay0_exp20_size23M_seed2/step_0005000.pt` (acc 0.792)
- d50 null: `checkpoints/arith_delay50_exp20_size23M_seed2/step_0005000.pt` (acc 0.626)
- lr_shift-treated: `checkpoints/arith_delay50_exp20_size23M_seed2_ivlrshift0at2501/step_0005000.pt` (acc 0.778)

So the seed-2 check = running diagnostic tooling on existing checkpoints only.
"Partial" is bounded by analysis time, not compute.

(Seed-1 triad, for reference: `..._seed1/` early 0.798; `..._seed1/` d50 null
0.524 canonical [recheck] / 0.506 pre-hardening; `..._seed1_ivlrshift0at2501/`
treated 0.752. Seed-3 triad also exists if escalation is needed.)

## Pre-registered read (fix meaning before seeing numbers)
- **Seed 2 verdict == seed 1 verdict** (both "repaired" OR both "new") → circuit
  identity is deficit-severity-INVARIANT → claim generalizes; report at n=2 with
  the deep(0.274)/shallow(0.166) span as the evidence it is not severity-driven.
- **Seed 2 verdict != seed 1 verdict** → circuit identity DEPENDS on deficit
  severity → itself a finding, and a hard stop on any single-seed claim →
  escalate to seed 3 for the gradient direction.
- **PRE-BANNED:** explaining away a seed-2 divergence as "too shallow to have a
  circuit to diagnose" without first confirming seed 2 acquired arith — it did
  (0.778 treated / 0.626 null, both well above floor).

## Unconditional (regardless of seed count)
Build tooling parameterized over `{seed, arm}` checkpoint paths and layer indices
— NEVER hardcoded to seed 1's paths. ~Zero build-time cost, and it is the precise
failure mode being guarded against. Seed count then stays a runtime flag, not a
rebuild.
