# Pre-registration — does the critical period survive a non-cosine schedule?

**Written 2026-08-11, before any WSD run existed.**
Batch: `scripts/wsd_schedule_batch.sh`. Schedule: `reopen.train.lr_at_step`,
`lr_schedule="wsd"`.

## 1. What is being closed

Every run in the paper uses warmup + cosine decay to 0.1×peak. The mechanism
claim generalises over *schedules* ("a property of the optimizer's decayed
learning-rate schedule"), but the evidence is from exactly one schedule shape.
If the effect is specific to cosine's curvature, the claim — and the title —
overreach.

## 2. Why this is a falsification test, not a robustness check

The two schedules differ precisely where the experiment intervenes:

| | LR at step 2500 (the d50 onset) | LR at 4000 |
|:--|--:|--:|
| cosine | 3.57e-4 (60% of peak, already decaying) | 1.18e-4 |
| **WSD** | **6.00e-4 (still at peak)** | 6.00e-4 (decay starts here) |

Under WSD the delayed skill window opens while the learning rate is still at its
maximum. If closure at d50 is mainly a *decayed-LR* phenomenon, the deficit
should largely disappear. So:

- **Deficit vanishes or shrinks sharply under WSD** → the schedule account is
  confirmed on an independent schedule, and it becomes a statement about decayed
  schedules generally rather than about cosine.
- **Deficit persists at ~0.21** → closure is *not* mainly the LR schedule.
  §4.4's central claim would need rewriting, and the cosine result would have to
  be re-read as coincidental alignment between the decay curve and the onset.
- **Deficit shrinks but survives** → both factors contribute; report the
  schedule as a major but partial cause, with the WSD deficit as the residual
  attributable to something else (entrenchment being the standing candidate).

All three outcomes are reportable. The second is the one that costs the paper
its headline, which is exactly why it must be run before submission rather than
discovered by a reviewer.

## 3. Design

Phase 1: WSD × {delay 0.0, delay 0.5} × seeds {1,2,3} = **6 runs**, ~25 min each
at 23M. Identical to the existing cosine arith runs in every other respect
(`--skill arith --skill-exposure-frac 0.20 --model-size 23M --total-steps 5000
--checkpoint-interval 100`). WSD shares cosine's peak (6e-4), floor (6e-5) and
warmup (300 steps); only the shape between them differs — a constant plateau,
then a linear decay over the final 20% of steps.

Runs occupy their own `run_id` namespace (`arith_delay50_exp20_wsd_size23M_...`),
so they cannot collide with or overwrite the cosine runs.

**Phase 2 is conditional and is not launched with phase 1.** If and only if a
deficit appears under WSD, the `lr_shift` arm is forked under WSD at n = 3. If
there is no deficit there is nothing to reopen, and running a reopening arm
against a null deficit would produce an uninterpretable number.

## 4. Comparators and decision rule — fixed before the data

Cosine reference, in hand at n = 3:

| | endpoint |
|:--|:--|
| cosine early anchor (d0) | 0.791 ± 0.007 |
| cosine delayed (d50) | 0.577 ± 0.051 |
| **cosine deficit** | **0.214 ± 0.055** |

Primary statistic: the **paired per-seed deficit** under WSD,
d_s = acc(d0, seed s) − acc(d50, seed s), reported as mean ± SD across the three
seeds, with the WSD d0 and d50 endpoints given as two separate numbers beside it
(never collapsed into a ratio, and never as percent-of-gap).

- **Deficit present under WSD**: paired deficit > 0 at all three seeds and mean
  deficit > 0.05.
- **Deficit substantially reduced**: WSD deficit mean < half the cosine deficit
  (< 0.107), i.e. outside cosine's per-seed range on the low side.
- **Deficit abolished**: paired deficit straddles zero in sign across seeds, or
  mean ≤ 0.05.

Comparison of the two deficits (WSD vs cosine) is Welch's t on the three paired
per-seed deficits per schedule. **The MDE is quoted with the result and the
result is read off the MDE, never off the p** — at n = 3 vs 3 with the observed
SDs, only large differences are detectable, so a non-significant difference
between schedules is reported as "consistent with, and not distinguishable
from" and never as "the same".

An important asymmetry, fixed here: the *presence* of a WSD deficit is
well-resolved (it is a paired within-seed contrast against a tight anchor); the
*equality* of the two deficits is not. Do not write "the deficit is the same
under both schedules."

**Multiplicity.** One primary test (WSD paired deficit ≠ 0) and one secondary
(WSD deficit vs cosine deficit). Holm within this family of two, declared
separately from the Detect, Reopen, n=3-arms and decomposition families.

**Perplexity.** Held-out perplexity is reported for all six runs. WSD trains at
peak LR far longer than cosine, so its absolute perplexity may differ from the
cosine runs; that is expected and is **not** evidence about the deficit, which is
a within-schedule contrast. Cross-schedule perplexity differences are descriptive
only.

## 5. Scope

One non-cosine schedule, one size (23M), one onset (d50), one skill
(arithmetic). A WSD result does not license "schedule-general" without
qualification; it licenses "replicates on a second, deliberately dissimilar
schedule". Linear decay was considered and dropped as too close to cosine to be
an independent test — a monotone decay with slightly different curvature would
have shared cosine's key property (already-decayed LR at the onset), which is
the very thing under test.
