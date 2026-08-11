# Representational probe — filler-gap dependency across delay (23M, seeds 1-3)

**Question (user idea #1):** every LD null so far (mean, dynamics, rate) is *behavioral*.
Does delay change what the model *encodes* about long-distance wh-licensing, even though
next-token accuracy is floored/flat in every condition? (Representation-precedes-behavior.)

**Method:** last-token linear probe (closed-form ridge, grouped 5-fold CV) on the pooled
BLiMP minimal pairs. Pooling with_gap (good=wh) + no_gap (good=that) **balances complementizer
identity across the grammaticality label**, so the probe cannot cheat off the wh/that token —
it must read whether the complementizer matches the gap (the licensing interaction itself),
off the representation. Read at the last token (post-gap) so LD items require the filler to
have been *maintained* across the long dependency. Guards: shuffled-label **selectivity**
control at the best layer; **local** paradigm as positive control; **seeds 1-3** per delay;
grouped CV by pair so surface lexis can't leak. No new training — reads step-5000 checkpoints.

## Results (chance = 0.5)

| delay | LD probe mean±SD | selectivity | local probe (pos. control) |
|------:|:----------------:|:-----------:|:--------------------------:|
| 0  | 0.6115 ± 0.0161 | +0.117 | 0.6535 |
| 25 | 0.5975 ± 0.0306 | +0.097 | 0.6452 |
| 50 | 0.5991 ± 0.0222 | +0.101 | 0.6528 |
| 75 | 0.5968 ± 0.0237 | +0.096 | 0.6488 |

Signal is broad across layers 3-7 (~0.57-0.63), peaking mid-to-late (layers 4-6) — not one
lucky layer.

### Three findings

1. **Filler-gap structure IS encoded** (contra "not learnable at all"). Pooled over all runs
   (n=12): probe **0.601 vs chance**, t=16.5, p<1e-8; **selectivity +0.103 vs shuffled**,
   t=12.0, p<1e-8. The positive control (local) reads higher still, 0.650 — so the probe works
   and the LD read is informative, not a dead probe.

2. **No representational critical period.** LD probe is flat across all four delays; the key
   contrast **delay0 (0.6115) vs delay50 (0.5991) is n.s. (Welch p=0.48)**. Delay does not
   degrade what the model encodes about the dependency.

3. **Representation > behavior, but timing-invariant.** Probe LD ≈ 0.60 sits *above* the
   behavioral LD floor (~0.51). So representation-precedes-behavior is real here — the model
   encodes more filler-gap structure than its output uses — but that gap is the SAME in early
   and delayed conditions. The behavioral floor is a **readout/usage bottleneck, not an
   encoding absence**, and neither the encoding nor the readout is modulated by timing.

**Caveat (honest):** best-layer selection (max over 8 layers) inflates absolute probe values,
but (a) it is applied identically to every condition, so the cross-condition CP contrast is
unbiased, and (b) the "encoded at all" claim rests on the selectivity control (shuffled labels
at the same selected layer → ~0.50), not the raw value.

## Verdict for the LD status question

A **fourth null** for the critical period — and a richer one than the behavioral nulls. It
does not merely say "no timing effect"; it says the LD structure is weakly but genuinely
encoded regardless of when syntax is introduced, the bottleneck is readout not representation,
and timing touches neither level. This *strengthens* "not robustly learnable in this
data/scale regime" and adds a real mechanistic nuance for the paper.

Artifacts: `summary.md` (auto), `per_run.csv`, `per_run_full.json` (per-layer curves),
`scripts/syntax_ld_probe.py`.
