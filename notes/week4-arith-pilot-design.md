# Week 4 Pilot Design — Arithmetic Reopening + LR-Segment Dissection

Status: DESIGN (2026-07-18). Arithmetic only — syntax Week-4 design is ON
HOLD until the seed 4-8 delay-0 acquisition-rate read lands. Numbers marked
PROPOSED are decision points, not decisions.

## Source run and common checkpoint

Everything forks from **`arith_delay50_exp20_size23M_seed1`** at
**step 2500** (= its skill onset; checkpoint file
`checkpoints/arith_delay50_exp20_size23M_seed1/step_0002500.pt`).

- Week 3 result: exact-match 0.798 (delay-0) → 0.506 (delay-50), gap ≈22 SE.
- d50 chosen over d25 (gap too small: 0.724) and d55 (adds nothing: 0.504).
- COMMON-CHECKPOINT RULE: every arm resumes from that one file via
  `scripts/run_intervention.py`, which reconstructs the config from the
  checkpoint itself and stamps `forked_from` into all downstream
  checkpoints. Record the file's sha256 in the launch log.
- Equal budget by construction: all arms (and the untouched source run,
  which serves as the **null arm for free**) run the same steps 2501-5000
  on the same post-2500 stream (seed-reconstructed; cram is the licensed
  exception). Recovery is read at step 5000 against two anchors measured
  at the same step: delayed-null (0.506) and delay-0 baseline (0.798).
- Why fork mid-run rather than extend past 5000: extending changes the LR
  schedule and total-token budget (unmatched vs both anchors); the
  extension variant is a grid-phase follow-up only if everything here
  shows zero recovery.

## Arms (seed 1, ~15 min each, ~1.5-2 h total)

| # | arm | command core (all: `--from-run arith_delay50_exp20_size23M_seed1 --fork-step 2500`) |
|---|-----|-----|
| 0 | null | — (the source run itself, already trained) |
| 1 | reset (embedding) | `--intervention reset` |
| 2 | reset + LR spike | `--intervention reset_spike` |
| 3 | LR spike alone | `--intervention spike` |
| 4 | freeze (only embedding trainable) | `--intervention freeze` |
| 5 | cram | `--intervention cram` (see headroom caveat) |
| 6 | LR shift (dissection) | `--intervention lr_shift --shift-source-start 0` |

Intervention step defaults to 2501 (fire at the first post-fork step =
re-exposure restart). PROPOSED parameters (single location:
`run_intervention.py` placeholders): spike peak 6e-4 (= original peak) over
300 steps (mirrors warmup length); freeze/cram windows 500 steps.

**Cram dose limitation — RESOLVED (2026-07-18): keep cram at d50, document
the dose limit; do NOT move it to d25.** The asymmetry and its handling are
explicit here and must appear in the Week 9 stats model, not be silently
baked in.

- Numbers: d50-exp20 stream density = exposure/(1−d) = 0.20/0.5 = 0.40.
  The 0.5 stream-dominance cap leaves a cram boost of ≤0.49 = **+22.5%
  relative**. Reset/spike/freeze/lr_shift have NO analogous dosing ceiling.
- Why not d25: d25 gives 83% relative headroom (density 0.267) BUT its
  Week-3 gap is only ~0.074 (exact-match 0.724 vs 0.798) — cram there would
  be reopening a near-absent deficit, a different and weaker question, and
  it would break the single-common-checkpoint symmetry (different source
  run, different fork step). Two wrongs, not a fix.
- Decision: all four Tier-1 interventions fork from the SAME d50@2500
  checkpoint. Cram stays in that symmetric set. Its dose limitation is
  handled in INTERPRETATION, not experiment structure:
  **STATS-DESIGN RULE (Week 9):** cram is NOT a peer of the other three in
  the recovery-percentage model. A null/weak cram result at d50 is
  uninformative about "does concentrated re-exposure reopen" in general —
  it only says "cram at the max density d50's already-dense stream allows
  (+22.5%) does not reopen." Record cram's achievable relative dose as a
  covariate/footnote; never read a d50 cram null as "the method fails."
- A properly-powered cram test needs a dedicated LOW-total-exposure design
  (a model that genuinely under-learned arithmetic, so there is dosing room
  AND a real gap) — a separate future experiment, out of scope for the d50
  reopening pilot.

## LR-segment dissection (why arm 6 + arm 2 vs 1 matter most)

Open confound from Week 3 (density-decoupling control): early-burst hit
0.778 at its window end vs 0.506 for d50 at the same density and window
length — read as "timing beats density", BUT the early window trains under
warmup+peak LR while the late window trains under the decayed tail
(6e-4 → 6e-5). "Timing" and "LR segment" are currently confounded.

- **Contrast A (arm 6 vs arm 0):** arm 6 replays the schedule segment
  [0, 2500) inside the late window — same weights, same data timing, early
  LR. If exact-match → ~0.77 the "critical period" is largely
  optimizer-schedule closure; if it stays ~0.51, LR is exonerated and
  representational/data-order entrenchment carries the effect. In-between
  = partial attribution; report the fraction of the gap closed.
- **Contrast B (arm 2 vs arm 1):** the pre-registered reset × LR confound
  split — reset-alone failing while reset+spike recovers means fresh
  weights starve under decayed LR (schedule plasticity), not that reset
  is the wrong lever.
- Complement (fresh run, optional phase 2): early window under the late
  segment — `train_baseline.py --skill arith --skill-exposure-frac 0.20
  --burst-end-frac 0.5 --lr-shift 0.0:0.5:0.5` (auto-suffixed). Predicted
  to fall toward 0.506 if LR drives the gap. Run only if Contrast A is
  positive and needs the converse.
- Caveat: arm 6 raises LR for ALL params late in training — general-LM
  cost is expected and is part of the answer (see ppl gate below); a
  "recovery" that wrecks val_ppl is forgetting, not reopening.

## Read-out (pre-register before launching)

Primary: final-checkpoint (step 5000) arith exact-match, full breakdown by
digit-count × carrying (`scripts/arith_breakdown.py`) — the Week 3 gap is
steeper at the hard end, so report per-band recovery, not just aggregate.

**Anchors (measured, seed 1, step 5000):** delay-0 baseline exact-match
0.798 @ val_ppl 48.19 (a); d50 null exact-match 0.506 @ val_ppl 50.83 (b).
Recovery gap = 0.292 exact-match; delay ppl cost b−a = **+2.64 ppl**.

- **Recovery gate (skill):** Recovery% = (arm − 0.506) / 0.292.
  - PRIMARY threshold (matches the operational-definition wording, "within
    90% of the early-exposure baseline"): arm exact-match ≥ 0.9 × 0.798 =
    **0.718** aggregate (= 72.6% of the gap closed).
  - Report the stricter recovery-fraction reading alongside: 90% of the
    GAP closed = 0.506 + 0.9×0.292 = **0.769**. State which an arm clears.
  - Per-band (digit×carry) recovery reported too — Week-3 gap is steeper at
    the hard end, so aggregate can mask a hard-band failure.
- **Forgetting gate (general LM), 3-number decomposition (locked rule):**
  (a) delay-0 val_ppl 48.19, (b) d50 null val_ppl 50.83, (c) arm val_ppl
  at 5000. Delay cost b−a = +2.64; **intervention cost = c−b**.
  - PROPOSED gate: c−b ≤ **+1.0 ppl** to qualify as reopening (≈38% of the
    2.64 delay cost, ~2% relative). c−b > +1.0 → flag forgetting-tradeoff
    regardless of skill recovery. c−b ≤ 0 (arm improves general LM) is fine.
  - All three thresholds (0.718 / 0.769 / +1.0) are PROPOSED and tunable —
    they are pre-registered here so they are fixed BEFORE any arm's number
    is seen, not chosen to fit results.
- Window-end reads only (recency rule from the burst control): all reads
  at step 5000 where every arm's exposure runs to the measurement point;
  cram's boost ends at ~3000 but its baseline 0.40 stream continues, so
  no washout asymmetry.
- Seed 1 only in the pilot; any arm crossing threshold gets seeds 2-3
  before being believed (Week 3 lesson: single-seed effects can be
  baseline artifacts).

## Validation gate (before any arm is launched)

No-op fork check (task #3): `--intervention lr_shift --shift-source-start
2501` makes the override equal the schedule, so the fork must reproduce
the source run's own continuation — final exact-match/val_ppl within the
same-horizon nondeterminism floor. This exercises fork-config
reconstruction, stream replay, and checkpointing end-to-end. If it fails,
nothing else launches.

**PASSED 2026-07-18.** Three-way max|d_val_loss| over post-fork ckpts:
source-vs-fork1 0.00230, source-vs-fork2 0.00196, and the pure-nondeterminism
control fork1-vs-fork2 0.00161 — all in one band, all peaking mid-run
(step 3200-3700) then converging (dvl@5000 ≤ 0.0007; val_ppl spread 0.037,
arith_acc spread 0.018 across all three). Fork-vs-source is NOT systematically
larger than fork-vs-fork, so the fork/resume/config-reconstruction path is
exonerated. NOTE: the pre-registered "~0.001" was the 240-step
validate_resume floor; the true 2500-step-horizon floor is ~0.0016-0.0023.
Use fork-vs-fork as the reference for any future longer-horizon resume check,
not the 240-step number.

**Stateful-intervention resume check (replay_fire) — PASSED 2026-07-18**
(`scripts/validate_intervention_resume.py --sensitivity`). The no-op fork
did NOT exercise replay_fire (stateless trigger, no mid-window kill); this
does. Vehicle = cram, killed+resumed mid-cram-window. PRIMARY gate is
DETERMINISTIC (stronger than a val_loss null-test): all 4 fixed trials'
realized token exposure is bit-identical to the straight runs (arith
1,536,000 / base 2,396,160), proving the loader-density state is rebuilt
exactly. Power control: 4 broken-replay trials (replay_fire forced off) all
diverge (arith 389,120) — the check provably detects a broken replay.
val_loss seam is a non-gating diagnostic (fixed ~0.012 ≈ ref floor 0.018).
Fixed a real CramIntervention resume-fragility (restore-density was cached
at at_step; now derived from trainer.streams) and an atomic-checkpoint race
(kill mid-torch.save corrupted the .pt; now temp-file + os.replace).

Intervention arms are cleared to launch (both resume paths validated).
