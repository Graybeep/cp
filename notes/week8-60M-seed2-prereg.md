# Pre-registration — 60M seed 2 circuit identity (replication of the scale effect)

Written 2026-07-26, BEFORE running `week8_circuit_identity.py --seed 2 --size 60M`.

## What is being tested

The Week-8 headline claim "**scale pushes toward reorganization independently of deficit
depth**" currently rests on a single 60M triad (seed 1: deficit 0.224, transfer 0.45,
control 2.2, Jaccard 0.09) contrasted against deficit-matched 23M models that show
repair (seeds 2/3: deficits 0.182/0.206, transfer 0.71/0.87).

n=1 → this is precisely the case CLAUDE.md's **control-arm-seed-depth rule** covers: a
single seed cannot distinguish "scale reorganizes" from "one seed happened to land low."

## Reference distributions (fixed now, from data already on disk)

- 23M repair band, point estimates: **0.71, 0.87**
- 23M repair band, bootstrap CIs over eval problems: **[0.665, 0.757]** (s2), **[0.834, 0.902]** (s3)
  → pooled lower edge of the repair regime = **0.665**
- 23M deep-deficit reorganized case: transfer **0.19** (deficit 0.292)
- 60M seed 1: transfer **0.45** (deficit 0.224)

## PRECONDITION — check before interpreting transfer at all

60M seed 2's deficit (A−B) must fall in the **moderate band ~0.15–0.25**, i.e. the range
where 23M shows repair. If 60M seed 2's deficit comes in **deep (≥0.28, seed-1-23M-like)**,
then a low transfer is confounded with deficit depth and CANNOT be read as a scale effect —
it would only be consistent with it. Report as confounded, not as replication.
Conversely if the deficit is near zero (no CP at this seed), the triad is uninformative
about identity and must be reported as such, not as "repair."

## Read rule (decided in advance)

Primary readout = **transfer ratio C/A**. Secondary = control ratio (must be ≥2.0 for any
"reorganization" reading — otherwise C is merely ablation-robust and low transfer is an
artifact). Jaccard is corroborative only (it did NOT discriminate at 23M: seed 2 had
Jaccard 0.09 *with* transfer 0.71).

Given the precondition holds:

- **REPLICATES (scale effect real, statable as a finding):** transfer **≤ 0.60**
  (clear of the 23M repair regime's lower bootstrap edge 0.665, with margin) AND
  control ≥ 2.0. Report as: scale-driven reorganization, n=2 at 60M, with both
  per-seed endpoints shown beside the 23M deficit-matched pair (never collapsed
  into a single ratio — third clause of the seed-depth rule).
- **FAILS TO REPLICATE (walk the claim back):** transfer **≥ 0.665** (inside the 23M
  repair regime). Then 60M seed 1 was a one-off draw; the write-up must be revised from
  "identity depends on scale" to "**suggestive, unreplicated** — one 60M seed reorganized,
  a second did not." Headline §60M bullet gets demoted out of the HEADLINE block.
- **AMBIGUOUS:** 0.60 < transfer < 0.665, or control < 2.0. Then n=2 is insufficient and
  **seed 3 at 60M is run** (checkpoints already on disk) before any claim either way.
  Escalation is pre-authorized — no re-decision after seeing the number.

## Reporting constraint

Report **absolute transfer endpoints per seed ± the bootstrap CI**, alongside the
deficit-matched 23M endpoints as separate numbers. No "X× more reorganized" ratio
(retired-metric rule). Deficit values quoted per seed, since deficit is seed-variable here.

## Order of work

1. This run (60M seed 2) — highest stakes, blocks the headline claim.
2. Seed 3 at 60M ONLY if the result is ambiguous per the rule above.
3. Seeds 2/3 activation-patching corroboration at 23M — lower-stakes refinement, after.
