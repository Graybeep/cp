# Week 4 — seeds 2-3 replication of the lr_shift reopening result

Ran 2026-07-21 (recovered after a mid-run power loss; see
`memory/week4-intervention-design-rules.md` for the corruption recovery).
Primary metric = arith exact-match accuracy at step 5000. Forgetting check =
general val_ppl (intervention cost c−b must be ≤ +1.0 ppl). PRIMARY skill gate
= arm ≥ 0.718 (90% of the delay-0 baseline 0.798, from the seed-1 anchors).

## Arithmetic d50 — null (baseline) vs lr_shift

Numbers independently reproduced by re-evaluating the on-disk step-5000
checkpoints (deterministic arith eval) — all four match metrics.csv to 6 d.p.,
incl. seed2's resumed-lineage null → not a resume artifact. Checkpoint
integrity separately verified (no stray .tmp, multi-position load, uniform
278.5MB).

| seed | null acc | lr_shift acc | lift | null ppl | lr_shift ppl | c−b (forget) | gate ≥0.718 |
|------|---------:|-------------:|-----:|---------:|-------------:|-------------:|:-----------:|
| 1*   | 0.524    | 0.752        | +0.228 | 50.75  | 49.86        | −0.89        | PASS |
| 2    | 0.626    | 0.778        | +0.152 | 48.58  | 48.38        | −0.20        | PASS |
| 3    | 0.582    | 0.772        | +0.190 | 47.70  | 47.90        | +0.20        | PASS |

\* seed-1 null = hardened-code recheck **0.524** (canonical; supersedes the
pre-hardening pilot 0.506, see `logs/week4_close/results.md` Item 1). Lift/gap
recomputed against 0.524. The lr_shift arm and gate are unaffected (gate = 90%
of the 0.798 early anchor, not null-derived).

**CORRECTED 2026-08-05 — seed-1 ppl columns were MIXED-LINEAGE.** The row above
originally read `null ppl 50.83 | lr_shift ppl ~48–50 | c−b ok`. Those two cells
came from *different models*: 0.524 is `…size23M_seed1_recheck`, but 50.83 is the
**superseded** pre-hardening `…size23M_seed1` (the run whose accuracy is 0.506) —
i.e. the accuracy had been corrected to the canonical lineage in 2026-07-21 and the
perplexity had not. Backbone §5 rule 2 (run_id *is* the model's identity) forbids
exactly this pairing. Canonical values, read from each run's own `metrics.csv` at
step 5000: null `…seed1_recheck` **50.747**, arm `…seed1_ivlrshift0at2501`
**49.858**, so **c − b = −0.889** — an *improvement*, not merely "ok", and the
largest of the three seeds. Superseded values retained in this note for provenance.
**No conclusion moves:** all three seeds sit far inside the +1.0 forgetting gate and
the direction of the 23M no-breach finding is unchanged (it strengthens slightly:
two of three seeds improve, and the third is +0.19). Found while sourcing §4 of the
draft; the paper draft already carries the canonical lineage.

- **lr_shift arm: mean 0.767, SD 0.014** (0.752 / 0.778 / 0.772) — remarkably
  tight across seeds.
- **null arm: mean 0.577, SD 0.051** (0.524 / 0.626 / 0.582) — highly
  seed-variable, consistent with the Week-3 baseline-variance caveat. (With the
  pre-hardening 0.506 it was 0.571 ± 0.061; correction changes no conclusion.)
- **All three lr_shift arms clear the 0.718 primary gate.** No forgetting
  (c−b = −0.89 / −0.20 / +0.20 across seeds 1/2/3, all ≪ +1.0; two of three
  improve — seed 1's value corrected 2026-08-05, see the lineage note above).

**The "+84% recovery" figure does NOT replicate — and shouldn't be the
headline.** Per-seed recovery-as-%-of-gap is 83% / 62% / 65% (seed-1 recomputed
against the canonical 0.524 null: (0.752−0.524)/(0.798−0.524); it was 84% at the
old 0.506 — still highest only because seed 1 had the lowest null → the biggest
gap to recover a fraction of). The attenuation is driven by the NULL rising
across seeds, NOT the treated arm falling (treated is flat at 0.767 ± 0.014).
Because the d50 gap is itself seed-variable, percent-of-gap is unstable by
construction. **Headline = the absolute claim: lr_shift reaches within 90% of
the early baseline (≥0.718) at n=3; the specific ~83% was seed-1's larger gap.** (See CLAUDE.md standing
rule "control-arm seed depth" / absolute-endpoint reporting.)

## Negative control — syntax d50, same lr_shift intervention (PER SEED)

Pre-registered CLEAN metric = continuation licensing interaction
(`scripts/gap_region_interaction.py`), same as the seed-1 Q2 read — NOT the
in-training `syntax_accuracy` proxy. Each seed has BOTH a null (untreated
syntax-d50 baseline) and an lr_shift arm, so the control is replicated at the
same seed depth as the arith arm — a single-seed control could not distinguish
"no generic effect" from "an effect that happens to be ~0 in one seed".

| seed | null local | lr_shift local | Δ local | null LD | lr_shift LD | Δ LD |
|------|-----------:|---------------:|--------:|--------:|------------:|-----:|
| 1    | 0.653      | 0.629          | −0.024  | 0.545   | 0.524       | −0.021 |
| 2    | 0.611      | 0.607          | −0.004  | 0.520   | 0.527       | +0.007 |
| 3    | 0.611      | 0.655          | +0.044  | 0.518   | 0.533       | +0.015 |

lr_shift on delayed syntax produces **sign-inconsistent noise around zero** at
every seed (Δ local ∈ [−0.024, +0.044]; Δ LD ∈ [−0.021, +0.015]; both means
≈ 0). NB the syntax null/lr_shift LD levels (~0.52–0.55) are syntax's OWN
weakly-learned delayed equilibrium — *above* the 0.500 prior-only chance, NOT
chance itself; the finding is the **absence of movement** between null and
lr_shift, not proximity to chance. By contrast the SAME intervention on arith
lifts **+0.246 / +0.152 / +0.190** per seed (exact-match units — a DIFFERENT
metric/scale from the continuation interaction, so this is a directional, not
magnitude, comparison). So the "generic late-LR effect" is **stably ≈ 0 across
all three seeds** while arith recovers at every seed → the arith recovery is
critical-period-specific, not a strong generic mechanism.

**Residual limitation (calibrates the eventual claim):** syntax (and vocab)
have no CP gap to reopen, so this control tests "lr_shift can't MANUFACTURE a
skill that was never entrenched," not "lr_shift won't reopen a
delayed-WITH-gap non-CP skill." The latter is untestable here — arith is the
only skill with a CP gap at 23M. A claim of the form "arith recovers via a
mechanism specific to CP closure" is supported; fully excluding "a generic
mechanism that happens to be strong on any skill that HAS a gap" would need a
second CP-gap skill, which this design does not have.

## Verdict

**The lr_shift reopening of the arith d50 critical period REPLICATES across
seeds 1-3.** Every arm crosses the primary gate, none forgets, and the
negative control is clean. Strikingly, the intervention pins accuracy at
~0.77 regardless of where the untreated null landed (0.506–0.626) — the
null-arm variance does not propagate to the treated arm, which strengthens
rather than weakens the finding.

Caveat: recovery% expressed against a fixed 0.292 gap is seed-dependent
because the null moves; report the absolute treated accuracy (0.767 ± 0.014)
and the per-seed lift alongside any single recovery-percentage figure.
