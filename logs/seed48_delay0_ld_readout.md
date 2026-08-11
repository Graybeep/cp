# Seed 1-8 delay-0 LD acquisition read-out (2026-07-18, corrected)

Continuation-scored wh-licensing interaction (gap_region_interaction.py),
final checkpoints (step 5000). Pre-registered shape rules fixed 2026-07-18
before any seed-4-8 number existed. **This version corrects two errors in
the first draft** (flagged by the user): (1) the "0.534-0.544 cluster"
description silently excluded seeds 2-3; (2) seed 1's data-version
provenance was not checked.

## Full 8-value delay-0 long-distance list (sorted), with gaps

| LD    | seed | gap from prev | data version |
|-------|------|---------------|--------------|
| 0.520 | 3    | —             | current syntax.bin |
| 0.522 | 2    | +0.002        | current |
| 0.534 | 8    | +0.012        | current |
| 0.537 | 5    | +0.003        | current |
| 0.540 | 7    | +0.003        | current |
| 0.541 | 6    | +0.001        | current |
| 0.544 | 4    | +0.003        | current |
| 0.586 | 1    | **+0.042**    | **PRE-DATES current syntax.bin (see below)** |

Local (same checkpoints, reference): s1 0.679, s2 0.619, s3 0.599,
s4 0.622, s5 0.628, s6 0.657, s7 0.638, s8 0.686.

## CORRECTION (2026-07-19): "confound" was an UNVERIFIED inference — now under empirical test

The section below argued from TIMESTAMPS that seed 1's 0.586 is data-version-
confounded and should be excluded. That was overstated as settled. It is an
inference, not a check, and it **conflicts with the possibility that the 21:01
v2→v1 revert regenerated pool v1 byte-identically** (augment_syntax.py is
seed-driven; a deterministic regen from unchanged code+corpus would make seed
1's 15:36 data identical in content to the current file regardless of write
time). No artifact ever confirmed byte-identity either way — there is no
archived copy of the 15:36 pool, and this is not a git repo, so a hash-compare
is impossible.

**RESOLVED 2026-07-19 — recheck REPRODUCES seed 1's high value; confound REFUTED.**
Re-ran seed 1 from scratch on the CURRENT syntax.bin (sha256 1eefb670…),
run_id `syntax_delay0_exp10_size23M_seed1_recheck`, same seed/config (seeding
derives only from --seed, not run_id — train.py:84 / data.py:57 — so the suffix
doesn't perturb the compare). Continuation-scored (gap_region_interaction.py,
step 5000): **LD 0.583** (vs original 0.586, Δ0.003 = within the nondeterminism
floor), local 0.665 (vs 0.679). with_gap/no_gap LD = 0.306/0.859 = a real
gap-licensing contrast, not a prior artifact.

Consequence: seed 1's ~0.585 is a GENUINE, REPRODUCIBLE high draw on the exact
file every other seed trained on. The timestamp "confound" is dead — whether or
not the 15:36 and 21:01 pool bytes were identical, the current data produces the
same high value, so seed 1 is a valid point in the current-data distribution.
**Seed 1 stays IN.** The verdict below ("exclude seed 1 → deprioritize") is
WRONG and is retracted; see the corrected verdict in the next section.

## CORRECTED VERDICT (2026-07-19): rule (a) — rare reproducible high mode → lean REPOSITION

Full 8-value delay-0 LD list, ALL on verified current data (seed 1 = recheck):
**0.520, 0.522, 0.534, 0.537, 0.540, 0.541, 0.544, 0.583.**
Shape: 7 seeds cluster 0.520–0.544 (mean 0.534, SD 0.009); ONE seed (seed 1)
sits at 0.583, +0.039 above the next-highest, and it is REPRODUCIBLE (0.586→0.583
across two independent trainings). That is the user's pre-registered **rule (a)**:
6–7 of 8 near the low level with 1 up at 0.58+ = a real-but-rare high-acquisition
mode → **favors REPOSITIONING the model-size axis** to test whether that mode
becomes the norm at larger scale, rather than deprioritizing syntax outright.
Caveat: it is 1 of 8 seeds; the mode is real (reproducible) but rare, so the
reposition is to INVESTIGATE scale-dependence, not a claim that LD is healthy at
23M. Final reposition-vs-deprioritize call is the user's; the evidence now leans
reposition. ("Possibly bimodal" is back on the table as a hypothesis the
model-size sweep would test — no longer dropped.)

--- everything below is the SUPERSEDED original timestamp argument, kept for audit ---

## [SUPERSEDED — see correction above] Seed 1 timestamp argument

Timestamps: seed 1's run artifacts (config.json 15:11, last checkpoint /
metrics 15:36) are all **Jul 16 ~15:36**. The current `syntax.bin` was
written **Jul 16 21:01** (the v2→v1 revert; augment_meta.json and the
archived logs/augment_meta_poolv2.json share that 21:01 mtime). Seeds 2-8
all ran Jul 17-18 on the 21:01 file. So seed 1 trained on whatever syntax
pool existed at 15:36 — a DIFFERENT file write, possibly different content.
Whether the 21:01 regeneration was byte-identical to what seed 1 used is
UNVERIFIED, and the project's own rule (augment_meta note: "Runs on the
pre-augment pool are not comparable with runs on this one") says to treat
a cross-regeneration run as non-comparable absent proof otherwise.

**Consequence:** seed 1's 0.586 is not a clean data point. The one value
that would have argued for a high-acquisition mode is exactly the one
confounded with data version.

## Honest distribution (the 7 seeds that definitely share current data)

Seeds 2-8, all on the 21:01 `syntax.bin`:

- Sorted: 0.520, 0.522, 0.534, 0.537, 0.540, 0.541, 0.544.
- Range 0.520-0.544 (0.024), mean 0.534, SD 0.0094.
- **No seed exceeds 0.544.** Largest internal gap is 0.012 (0.522→0.534);
  the values are moderately dispersed, not a tight cluster, but show no
  second mode and no high tail.
- Visible sub-structure (seeds 2-3 at 0.520-0.522 vs seeds 4-8 at
  0.534-0.544) is consistent with seed variance at n=2 vs n=5 on identical
  data — do not over-read it; it is NOT a data-cohort effect (all seven
  share the 21:01 file).

## Verdict: DEPRIORITIZE syntax LD as a Detect skill at 23M

Cleaner than "lone outlier." Among the seven seeds on consistent current
data, delay-0 LD acquisition sits at 0.52-0.54 (mean 0.534) against a
0.500 prior-only floor — thin, and with a ceiling of 0.544. There is no
reproduced high-acquisition mode: the only >0.544 value (seed 1, 0.586)
is data-version-confounded. Delayed runs sit ~0.52 (Week 3), so on the
consistent-data seeds there is essentially **no delay headroom for the
Detect axis to resolve** — the delay gap the study needs is inside the
seed-noise band.

Mapping to the pre-registered rules: rule (a) (real-but-rare high mode →
reposition) is NOT supported — the high draw did not reproduce in 5 fresh
seeds AND is confounded. "Possibly bimodal" is DROPPED (one confounded
point is not a mode). The outcome is rule (c)'s decision (deprioritize),
reached via the cleaner argument that no consistent-data seed shows the
high mode at all.

Final reposition-vs-deprioritize call is the user's; the evidence points
firmly to deprioritize.

## Loose end worth one cheap check if syntax is ever revived

Confirm whether the 21:01 `syntax.bin` regeneration was byte-identical to
seed 1's 15:36 version (re-run seed 1 on the current file, or hash-compare
if the old file is recoverable). If identical, seed 1 rejoins the pool as
a genuine 0.586 high draw (still 1/8, still argues deprioritize). If
different, seed 1 is correctly excluded. Either way the decision holds; the
check only affects how seed 1 is described in the writeup.
