# Consolidated status — 2026-07-24

Single roll-up of the work since the Week-4 pilot: infrastructure hardening, the
Week-4 close-out, the syntax size-sweep, the scope-gap decision it forced, and the
arithmetic scale-check arc (A–D) that closed it. Each section points to the detailed
log/note; this is the map, not a replacement for them.

**One-line state:** Detect and Reopen for **arithmetic** are established and
**scale-robust** across 11M/23M/60M; **syntax long-distance** was dropped from the
Weeks 5-7 grid after a 72-run size-sweep disconfirm. Arithmetic now carries the Detect
axis. Next build is Week-8 circuit diagnosis (seed-scope already decided).

---

## 1. Infrastructure hardening — the two bug fixes

Both surfaced during Week-4 resume validation and both affect the **whole** Weeks 5-7
grid, not just interventions. Detail: `memory/week4-intervention-design-rules`.

- **Atomic-checkpoint save (BUG-1).** `save_checkpoint` wrote `torch.save` straight to
  the final path, so a supervisor launch-timeout kill mid-write left a **truncated**
  `step_*.pt` that `latest_step()` still counted → next resume crashed ("failed finding
  central directory"). Fixed in `reopen/checkpoint.py`: write to `<name>.pt.tmp` (not
  matched by the `step_*.pt` glob), then `os.replace()` (atomic rename) — a kill now
  leaves either the previous complete checkpoint or the new one, never a partial. Lives
  in the single shared save path (`reopen/checkpoint.py:save_checkpoint`), so it covers
  every run type. Failure mode is **fail-loud** (zip central dir is written last →
  torch.load never returns loadable-but-bad tensors → any checkpoint that ever loaded is
  clean by construction; no retroactive integrity sweep needed).
  - *Addendum (power loss ≠ process kill):* `os.replace` is atomic against a process
    kill but **not** against sudden power loss without fsync — a same-size but
    unflushed newest `step_*.pt` fails the same fail-loud way. Recovery procedure is
    known and one-command (load-check newest, `rm` if it fails, relaunch the idempotent
    supervised batch). The metrics.csv sidecar is likewise power-loss-corruptible and
    `Trainer._resume` was hardened to survive a blank/headerless CSV.

- **replay_fire on stateful interventions.** Triggers whose effect lives **outside** the
  checkpoint (Cram/EarlyBurst density, Freeze `requires_grad`) now refire during
  `Trainer._resume`'s stream replay; before this, a killed-and-resumed cram/burst/freeze
  run silently rebuilt the **wrong** stream/grad state. Reset (weights already in the
  checkpoint) and LR overrides (recomputed per in-window step) correctly do **not**
  refire. Validated deterministically (`validate_intervention_resume.py --sensitivity`):
  realized token exposure is bit-identical to the straight runs, and a forced-broken
  replay provably diverges (~25% exposure) — so the check detects a broken replay rather
  than passing by luck.

Together these make every long/overnight run under `run_supervised.py` kill- and
resume-safe, which is the precondition for trusting the n=3 grid numbers below.

---

## 2. Week-4 close-out — arithmetic reopening at 23M (n=3)

`memory/week4-arith-pilot-results`; logs `logs/week4_arith_pilot/`, `logs/week4_close/`.

Six arms forked from the **identical** common checkpoint `arith_delay50_exp20_size23M_
seed1@2500` (common-checkpoint rule). Verdict at n=3:

- **Only `lr_shift` reopens.** lr_shift 0.752 / 0.778 / 0.772 = **0.767 ± 0.014**, all
  clear the 0.718 gate; general ppl did not degrade. The other three Tier-1 methods
  fail: reset 0.586 (forgetting flag), reset+spike 0.576, spike 0.536, freeze 0.332
  (harmful), cram 0.574 (dose-capped — not a fair peer at d50; see below).
- **Mechanism (Contrast A positive):** the d50 arithmetic critical period is largely
  **LR-schedule closure, not representational entrenchment**. `cram+lr_shift` n=3
  (0.746 / 0.794 / 0.804 ≈ lr_shift ≫ cram-alone) confirms LR-starvation: cram's extra
  data was useful but LR-starved like everything else at d50.
- **Prediction #2 (reset best) NOT borne out.** Reset was predicted strongest; it
  flagged forgetting instead.
- **Secondary finding — CP severity is seed-variable.** Early anchors are tight
  (0.791 ± 0.007) but nulls vary (0.577 ± 0.051; seed-1 canonical 0.524) → deficit
  0.214 ± 0.055, driven by the *delayed* endpoint. Seed 1 is the deepest-hit on both
  readings — which is why it is a poor sole basis for any downstream claim (see §5, §6).
- Syntax negative control at 0.52 → the reopening is arithmetic-specific.
- Reconciliation: the seed-1 low null (0.524) was reproduced from scratch under hardened
  code (genuinely low, not a resume artifact); no conclusion changed.

**Status: Week 4 CLOSED at n=3.**

---

## 3. Syntax size-sweep — DISCONFIRM

`memory/syntax-size-sweep-preregistration`; `notes/week5-syntax-size-sweep-
preregistration.md`; `scripts/szsweep_gate.py`; `logs/szsweep_eval/gate_result.json`.

The Week-3 pilot had left one live hypothesis: syntax long-distance (LD) acquisition
might be a **rate** (P(high) ≈ 12.5% at 23M — one high seed in eight), which a larger
model could turn into the norm ("reposition the model-size axis"). Pre-registered as a
compound gate (Cochran-Armitage one-sided trend p<0.05 robust across thresholds AND
top-size floor ≥ 0.40), powered at 24 seeds/size, with the lone high seed pre-banned as
confirmation.

**Result (all 72 runs scored):** mean LD rises monotonically
**0.528 → 0.538 → 0.552** across ~11.5 / 23.2 / 60.3M — but clears the gate **only** at
the permissive threshold 0.55, and **fails** FLOOR@0.56 (top-size 0.208 < 0.40) and
TREND@0.57 (p = 0.074). The data-derived threshold 0.559 validated the 0.56 rule.
Verdict: **DISCONFIRM ("weak size sensitivity")** — sub-floor, not reposition.

**Consequence: syntax LD is DROPPED from the Weeks 5-7 grid.**

---

## 4. The scope-gap decision

Dropping syntax LD created a **scale-asymmetry**: syntax got a full 3-size sweep before
being cut, but arithmetic Detect/Reopen had only ever been measured at **23M**. Leaving
that asymmetry would mean the surviving Detect skill rests on a single model size while
the dropped one was held to a 3-size bar — an uneven evidentiary standard.

**Decision:** run an arithmetic **scale-check at 11M and 60M** (matching the sweep's
sizes) to test whether the 23M Detect gap and lr_shift reopening hold across scale,
before the finding carries the grid alone. Pre-registered with numeric go/no-go rules
(no eyeballing): `notes/week5-7-arith-scale-check-preregistration.md`. Zero new training
code — same `run_supervised` + `run_intervention` paths, only `--model-size` differs.

This is the bridge from §3 to §5.

---

## 5. Arithmetic scale-check arc (A–D) — RESULT

Ran as a seed-1 probe (2026-07-23) then n=3 follow-ups (2026-07-24). Read-out
`logs/week5-7_arith_scale/followups_readout.md`; mechanical gate
`scripts/arith_scale_gate.py` → `gate_result.json`; `memory/week5-7-arith-scale-check`.

**(A) Detect is scale-robust.** 60M gap **0.229** (d0 0.829 ± 0.027 / d50 0.600 ± 0.040),
one-sided Welch **p = 0.0010** → CP confirmed, matching 23M's 0.214 across a 2.6× size
step. At 11M the early anchor itself is below the 0.60 floor (d0 0.556) → Rule-1
**FLOOR/STOP**: read as a capacity/budget boundary (the anchor never plateaued, LR-
starved under standard cosine at this size), **not** a "CP absent" timing claim.

**(B) Reopen skill-recovery is scale-invariant.** 60M lr_shift **0.796 ± 0.022** clears
all three Rule-2 conditions and lands **in the same-endpoint band [0.722, 0.812]**,
overlapping 23M's 0.767 → the *recovered endpoint is scale-invariant* (extends Week-4's
"pins ~0.77"). **However** the forgetting gate trips at n=3 (c−b = **+1.67**, all seeds).

**(C) That breach is BOUNDED-ARTIFACT, not clean forgetting.** The drift control (no-op
fork, Δppl −0.012 / Δacc +0.011) rules out fork mechanics — the +1.67 is intervention-
induced. But the LR trace shows the arm is measured **mid-anneal at 6× the null's
terminal LR** (`ShiftedLRWindow(source_start=0)` replays the early schedule, so at step
5000 the arm sits at LR 3.57e-4 vs the null's annealed 6.0e-5). At **matched LR**, n=3,
the arm is **ahead of the null by 2.48 ± 0.24 ppl** — a lead that *exceeds* the 1.67
breach. So any genuine residual cost is bounded above by the breach, with the evidence
leaning toward the low (near-zero) end. **No specific extrapolated endpoint is quoted**
— only the arm's *starting position* for its remaining anneal is measured, not the
*slope*; a same-slope projection would imply the intervention slightly *outperforms* the
standard schedule, an interesting-if-true claim the data cannot yet support. Collapsing
this to a point estimate needs the arm extended past 5000 — a **non-trivial** build
(naive `--total-steps` is invalid: `total_steps` is in the cosine denominator and resume
with a changed value is asserted against; the cheap null-extension is uninformative
because the null is already annealed). Flagged as the single confirmatory build,
**worthwhile but non-blocking** given how clearly the matched-LR evidence already points.

**(D) 11M lr_shift is exposure-alignment reopening (H3), ~70%.** The discriminator
(apply lr_shift to the *early* run) shows lr_shift lifts the delayed run +0.389 but the
early run only +0.115 → generic (H2) +0.115 + alignment (H3) +0.274 = 0.389. So ~70% of
the effect requires the delayed-exposure profile (high LR replayed over the arith
window), ~30% is a generic optimizer boost. The gain is exposure-timing-specific, not
number inflation; the 11M FLOOR/STOP on the pre-registered cell still stands (this was
a separately-labelled exploratory follow-up).

**Two analysis-tooling corrections made during this arc, both fixed at the tooling level
(not hand-patched once):**
1. `arith_scale_gate.py`'s Welch p-value had an incomplete-beta symmetry bug (printed
   0.499 for t=8.23); fixed and cross-checked against scipy (0.00103). Detect was on the
   gap ≥ 0.15 branch, so the p-value was non-gating regardless.
2. The H2/H3 discriminator initially compared the two *boosted endpoints* (wrong
   quantity, 0.093, which flipped the verdict to H2); the correct test is lr_shift's lift
   over each arm's *own null* (0.115 + 0.274 = 0.389, reconciling), giving H3-dominant.
   The self-correction is documented with its *why* so the wrong reasoning can't be
   reintroduced.

---

## 6. Running discipline (why the numbers are trustworthy)

Threaded through all of the above, and worth stating once:

- **n=3 before any claim gets trusted** — the same bar applied to resume validation, the
  deficit-variability check, the negative control, and (this arc) the matched-LR
  horizon-artifact reading (not accepted on the favorable single-seed trace until it
  replicated at n=3, lead 2.48 ± 0.24).
- **Single-seed traps named, not ignored** — seed 1 is atypical twice (deepest deficit
  AND the Week-4 fork source), which is exactly why the Week-8 seed-scope was decided
  *against* seed-1-only (see §7).
- **Absolute endpoints ± SD over percent-of-gap**, because the null is itself
  seed-variable — a percent-of-gap denominator that moves cannot replicate.
- **Shortcuts checked for soundness before use** — the `--total-steps` cheap path was
  dropped once it turned out structurally invalid rather than forced; the null-extension
  fallback was dropped once shown uninformative.

---

## 7. Open threads / next

- **Week-8 circuit diagnosis** (`memory/week8-diagnosis-seed-scope`;
  `notes/week8-diagnosis-seed-scope.md`): seed-scope already decided — full battery
  (activation patching + ablation + CKA + pruning) on seed 1, plus a **partial
  confirmatory** circuit-identity pass on seed 2 (shallowest deficit → maximal contrast;
  triad already on disk). Tooling must be **seed-parametric**, never hardcoded to seed 1.
  This is the next build.
- **Arm-extension** (from §5C): the only run that would collapse the 60M forgetting
  question from BOUNDED-ARTIFACT to a point estimate. Non-trivial to wire; non-blocking
  for the Weeks 5-7 write-up.
