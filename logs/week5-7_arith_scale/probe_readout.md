# Week 5-7 arith scale-check — PROBE read-out (seed 1, n=1)

**COMPLETE — all 6 probe arms finished 2026-07-23 13:32.**

## HEADLINE (n=1, both sizes)

| size | d0 | d50 | lr_shift | gap | lift | ppl cost (c−b) | Rule 1 |
|---|---|---|---|---|---|---|---|
| **11M** | 0.556 | 0.304 | 0.726 | 0.252 | 0.422 | +0.81 ✓ | **FLOOR → STOP** |
| **23M** (n=3 ref) | 0.791 | 0.577 | 0.767 | 0.214 | 0.190 | improved ✓ | (established) |
| **60M** | 0.798 | 0.574 | 0.796 | 0.224 | 0.222 | **+1.79 ✗ FLAG** | **CP present → GO fill** |

**60M reproduces 23M almost exactly across a 2.6× size difference**: d0 0.798 vs
0.791, d50 0.574 vs 0.577, gap 0.224 vs 0.214. The Detect finding looks
strikingly scale-stable.

**But 60M breaches the forgetting gate.** Intervention cost c−b (Week-4 locked
3-number decomposition: a = d0 ppl, b = d50 null ppl, c = arm ppl @5000) =
43.027 − 41.237 = **+1.790 > +1.0**. Per the CLAUDE.md operational definition
this is **forgetting, not reopening**, regardless of 2a–2c. It holds under both
readings, so it is not an artifact of using an absolute threshold at a different
ppl scale:
- absolute: +1.790 vs gate +1.0
- relative: **4.34%** vs the gate's ~1.97% (at 23M's 50.83 base) — ~2.2× over.

Skill recovery at 60M is otherwise essentially complete: 0.796 vs its own anchor
0.798 = **99.7%**, and 2b (≥ 0.90 × 0.798 = 0.718) would pass comfortably.
So the 60M n=1 picture is: **full skill recovery bought with a real general-LM
cost.** This is exactly the tradeoff the two-measurement rule exists to catch.

*Calibration note (flagged, NOT acted on):* the +1.0 gate was justified at 23M as
"≈38% of the 2.64 delay cost." At 60M the delay cost b−a is **+0.016** — there is
essentially no general-LM delay cost at this size — so the gate's original
relative justification does not transfer. The gate is nonetheless applied as
written (absolute +1.0), and the breach survives the relative reading anyway.
**No threshold is being retuned after seeing data.**

*Band note:* 60M lr_shift 0.796 sits inside the same-endpoint band
[0.722, 0.812], near its top. The pre-registered mechanical-consistency clause
fired exactly as written: sub-case (a) requires d0 > 0.802, and d0 = **0.798**,
so (a) is unreachable and an in-band endpoint simply passes 2b = full threshold
generalization. At n=3 the mean could still exit the band upward → sub-case (c)
"endpoint scales with capacity." **No band claim at n=1** (band is defined on an
n=3 mean with an SD ≤ 0.045 guard).

---

*(Sections below written 2026-07-23 02:30, while 60M d50 was still running.)*
Rules applied are those pre-registered in
`notes/week5-7-arith-scale-check-preregistration.md` (all fixed before any run).
Metric = `arith_accuracy` @ step 5000 from `logs/<run_id>/metrics.csv`.

## Numbers in hand

| size | arm | arith_acc @5000 | val_ppl |
|---|---|---|---|
| 11M | d0 (early anchor) | **0.556** | 59.84 |
| 11M | d50 (delayed null) | **0.304** | 59.54 |
| 11M | lr_shift (fork d50@2500) | **0.726** | 60.35 |
| 60M | d0 (early anchor) | **0.798** | 41.22 |
| 60M | d50 | *running* | — |
| 60M | lr_shift | *queued* | — |

11M gap (d0 − d50) = **0.252**; 11M lift (lr_shift − d50) = **0.422**.
11M forgetting check: 60.35 − 59.54 = **+0.81 ≤ +1.0** ✓ (no forgetting).

## 11M — mechanical verdict, and why its INTERPRETATION does not survive

**Rule 1 verdict: d0 = 0.556 < 0.60 → FLOOR → STOP at 11M, do not fill.**
The floor row takes precedence over the gap row, which would otherwise have said
GO (gap 0.252 ≥ 0.10). That is the rule as written and it is honored.

**But the pre-registered *reading* of a floor failure is empirically refuted at
11M, by 11M's own data.** The pre-registration phrases a floor failure as a
capacity boundary — "the model cannot learn the skill at all at this size,
independent of when it is introduced." At 11M the lr_shift arm, **same size,
same 60.3M→11M parameter count, same 5000-step budget, same 0.20 exposure**,
reached **0.726**. Capacity at 11M therefore permits at least 0.726. The
capacity claim is an "impossible above X" claim and a single counterexample
refutes it — this does not require n=3.

So 11M's d0 shortfall is **not** a capacity ceiling. It is an LR-schedule /
budget effect, and the pre-registered verbatim floor phrasing **must not be used
for 11M**. Two independent signs point the same way:
1. **OVERSHOOT fired.** lr_shift (0.726) exceeds its own early anchor d0 (0.556)
   by **+0.170**. This is the sub-clause pre-committed 2026-07-23 *before* any of
   these numbers existed; it fired on the first size measured.
2. **d0 never plateaued.** 11M d0 runs 0.200 → 0.508 → 0.556 at steps
   2500/3750/5000, still climbing at the horizon, where 23M plateaus by ~4000.

Read together with Week 4 (d50 arith closure is largely LR-SCHEDULE closure, not
entrenchment): at 11M the *early-exposure anchor itself* appears LR-starved under
the standard cosine decay. That makes the 11M "gap" a comparison between two
differently-schedule-limited runs, which is a distinct methodological problem
from the capacity story the floor rule assumes.

Band note: 0.726 falls inside the same-endpoint band [0.722, 0.812] — but the
band is defined on an **n=3 mean** with an SD ≤ 0.045 guard, neither of which
exists at n=1. **No band claim is made.**

## Open decision (not taken unilaterally)
Rule 1 STOPs 11M, but the stop's stated justification is refuted. Options:
- **(a)** Honor the STOP for the pre-registered claim structure (no 11M
  generalization claim of any tier), and report the OVERSHOOT as a documented
  anomaly at n=1.
- **(b)** Deviate: fill 11M to n=3 anyway (~0.9 h), recorded explicitly as a
  post-hoc deviation from Rule 1.
- **(c)** Recommended: (a) for the pre-registered cell, PLUS n=3 on the 11M
  OVERSHOOT as a **separately labelled exploratory follow-up** — keeps the
  pre-registration clean while not discarding an existence-proof-grade result.

## Process note
TWO background driver processes have been killed externally (not by me):
- `bkyllapq6` (probe driver) killed ~02:26, after 60M d0 completed and just as
  60M d50 started.
- `bdf8ldo9f` (watcher that would have fired the 60M lr_shift fork on d50
  completion) killed ~03:00, before it could fire. Empty output, no launch.

**The training runs themselves survive these kills** — they are orphaned from the
bash driver but keep running. 60M d50 was at step 3750 (arith 0.582 in-flight) at
03:01 with 12 live worker processes, ETA ~03:12 for step 5000. No checkpoint
corruption risk (atomic save + fail-loud, BUG-1 resolved).

**Remaining work: the 60M lr_shift fork only.** It cannot start until d50 writes
`step_0005000.pt`. Given two consecutive external kills, no third background
task was launched unilaterally. To fire it once d50 finishes:
`bash scripts/week5-7_arith_scale_check_batch.sh probe`
(idempotent — skips all four completed runs, executes only the fork, ~22 min).
