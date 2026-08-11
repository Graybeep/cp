# Pre-registration — Weeks 5–7 DETECT grid (arithmetic-carried)

Status: **DRAFT for review (2026-07-21)**. Numbers marked **PROPOSED** are
tunable-before-first-number (project standing rule); once the first grid eval
number exists they lock. Same discipline as the syntax size-sweep pre-reg
(`week5-syntax-size-sweep-preregistration.md`): confirm AND disconfirm criteria
fixed in advance, in both directions.

This document covers the **Detect** half of Weeks 5–7 only. The **Reopen** half
(intervention arms × the delay cells that show a gap) is gated on the Week-4
pilot and pre-registered separately.

---

## What changed upstream (why this is arithmetic-only)

- **Syntax LD DROPPED from the grid** — the delay-0 size-sweep DISCONFIRMED
  reposition (2026-07-21: monotone but sub-floor, "weak size sensitivity";
  `logs/szsweep_eval/gate_result.json`). Per its pre-registered gate, syntax LD
  does not earn a delay×size slot. Arithmetic carries Detect.
- **Arithmetic shows the (provisional) graded critical period** — Week-3 pilot:
  exact-match 0.798 (d0) → 0.724 (d25) → 0.506 (d50), ~22 SE, graded and
  steeper at the hard end (4d-carry).

**PROVISIONAL flag (must be surfaced, not buried):** the arith timing effect is
*provisional pending the Week-4 LR-segment dissection*. The early-burst control
showed a matched-density window at the START reaches ~0.778 vs the late window's
0.506 (timing, not density) — BUT both windows sit on different cosine-LR
segments, and LR decay is itself a candidate closure mechanism in this project.
Detect scale-up establishes the *gap's shape across size×delay*; the *mechanism*
(timing vs. LR-schedule) is a Week-4 question. This grid does not resolve it and
must not claim to.

---

## Sequencing decision (open — needs a call before launch)

Per the CLAUDE.md week plan, **Week 4 (Reopen pilot) precedes Weeks 5–7.** The
Week-4 arith intervention wiring is BUILT and no-op-validated, but the pilot
arms (reset / LR-spike / freeze / cram) have **not been run** (only validation
runs exist in `checkpoints/`). Two coherent orders:

- **(A) Detect grid first.** Detect needs no interventions; it is independent of
  Week-4 outcomes and produces the effect-size map that tells Reopen which cells
  to target. The provisional LR caveat is carried as a documented limitation.
- **(B) Week-4 Reopen pilot first** (strict schedule order), then this grid. The
  LR dissection lands before the grid, so the grid's arith result is reported
  clean rather than provisional.

Recommendation: **(A)** — Detect and Week-4 are parallelizable, and the grid's
per-cell gap map is the input Reopen needs anyway; the LR caveat is a reporting
footnote, not a blocker. Final call is the user's.

---

## Grid-boundary prerequisites (block the FIRST grid run; all break
## comparability with pilot numbers → new run_suffix `grid`)

1. **Stratified arithmetic eval + train.bin regen.** Current eval per-cell n's
   are incidental (easy cell n=63, a power hole separate from ceiling
   saturation). Rebuild: quota ≥150 per (digits × carry) cell (feasible — even
   2d-no-carry has 1,980 distinct problems), regenerate train.bin with the new
   larger exclusion set, all grid runs train on the new train.bin. Pilot↔grid
   numbers then not directly comparable (same rule as vocab.bin versions).
2. **Add a ~30M size tag** to `MODEL_SIZES` (registry currently 11M/23M/60M;
   60M is the size-sweep exception and stays OUT of this grid — the Detect size
   axis is locked to 10–30M). VRAM: 60M already fit in 3.06 GB reserved, so
   ~30M is trivially safe; confirm with `scripts/vram_probe.py` anyway
   (verify-don't-infer).
3. **Early-burst density-control plumbing.** `StreamSpec` has `onset_step` but
   no `end_step`; either add `end_step` (small feature) or use the existing cram
   seam (`MixedLoader.set_density(name, 0.0)` fired at step W). Post-burst
   density MUST be exactly 0 (a nonzero tail reintroduces the confound). Reads
   are **window-end vs window-end** (the pilot's recency confound: any condition
   whose exposure doesn't run to the measurement point is recency-biased).

---

## Design axes (Detect)

| axis | PROPOSED levels | notes |
|---|---|---|
| model size | **{11M, 23M, 30M}** | locked 10–30M range; 60M excluded |
| delay onset | **{0, 0.15, 0.30, 0.45, 0.55}** (5) | exp20 caps delay<0.60 (density<0.5); densities 0.20/0.235/0.286/0.364/0.444 |
| skill | **arithmetic (exp20)** + **vocab-frequency (negative control)** | syntax dropped |
| seeds | **PROPOSED 3** (see power note) | baseline-variance-aware read required |

- **Matched total exposure** across every delay cell (~16.38M arith skill
  tokens), verified per run as in the pilot — only *timing* differs.
- **Negative control (vocab-frequency)** runs the SAME delay schedule. It must
  show NO delay effect; a delay gap there flags a method problem, not the
  hypothesis (CLAUDE.md). Use log-prob not accuracy (vocab metric floors).
- Full arith grid = 3 sizes × 5 delays × 3 seeds = **45 runs**; vocab control
  ×(subset of delays or full) adds ~15–45. At ~3–6 min/run (size-dependent),
  batchable over ~3 nights via the same supervised/locked batch harness as the
  size-sweep.

### Seeds / power note

Arith's *within-run* effect is huge (~22 SE at n≈1000 eval), but the grid's
inferential unit is *between-seed* variance in final accuracy, which the pilot
never measured for arith (only seed 1 spanned the delays). **Pre-registered
interim check** (mirrors the vocab exit gate): after the delay-0 cells of all
three sizes finish (9 runs at 3 seeds), estimate between-seed SD; if it is
small relative to the pilot's 0.29 delay gap, 3 seeds suffice; if comparable,
add seeds before trusting any per-cell null. This keeps a null baseline-variance-
aware, the exact failure that produced the syntax seed-1 artifact.

---

## CONFIRM the critical period — pre-committed (arithmetic)

The operational definition (CLAUDE.md): a critical period ending at T exists if
late-onset models reach significantly lower final accuracy than early-onset,
*with matched total exposure*, AND the gap **persists after extended fine-tuning
on the skill alone**.

1. **Delay main effect.** Mixed-effects regression
   `final_accuracy ~ delay × model_size + (1|seed)`; critical-period signal =
   delay coefficient significantly **negative** (later onset → lower accuracy),
   p<0.05, holding after the baseline-variance check above.
2. **Persistence.** For the delay cells showing a gap, an extended skill-only
   fine-tuning tail (equal budget) must NOT close it to within noise — otherwise
   it is a "slow to learn," not a closed window. (Pre-register tail length =
   PROPOSED 2500 steps skill-only; window-end read.)
3. **Difficulty gradient (secondary, non-gating).** Per-(digits×carry) analysis;
   the pilot's "degrades everything, worse toward the hard end" replicates iff
   the hard cells (3–4 digit carry) show a larger delay drop than easy — now on
   the stratified eval so the easy cell isn't n=63.

## DISCONFIRM — the symmetric pre-commitment

- **Flat in delay:** delay coefficient n.s. and per-size CIs overlap across
  delays → no critical period at 10–30M → arith joins syntax as "no detectable
  Detect gap at this scale," and the study's Detect claim rests on neither skill
  (a real, reportable null with the scale caveat).
- **Gap closes under fine-tuning:** delay effect present at the checkpoint but
  erased by the skill-only tail → "delayed acquisition, not a closed window" —
  reported as such, NOT as a critical period.
- **Negative-control contamination:** if vocab-frequency ALSO shows a delay gap
  → the method manufactures delay effects → HALT and diagnose before any Detect
  claim (this overrides a positive arith result).

## Timing-vs-density (mandatory disentangling analysis, alongside)

Headline framing = delay-fraction (paper convention). Reported alongside, always:
(a) density-matched re-analysis (density = exposure/(1−delay)); (b) the
early-burst window-end control per size, separating lateness from burst density.
The confound is the recurring Pitfall-#1 structural risk (forgetting vs. genuine
effect) — write it up once as one risk surfacing in multiple places, not as
one-offs.

---

## Knobs tunable-before-first-number (then locked)

Model sizes {11,23,30M}, delays {0,.15,.30,.45,.55}, seeds (3, + interim power
check), skill-only tail length (2500), stratified-cell quota (≥150), whether the
vocab control runs the full delay set or a subset. Size range ceiling (30M) and
skill (arith, syntax dropped) are RESOLVED, not knobs. Any change happens before
the first grid eval number and is recorded with a reason.
