# Week 4 close-out — three residual items (ran 2026-07-21, seed grid n=3)

All arith numbers = exact-match at step 5000. Reporting per the CLAUDE.md rules
(absolute endpoints ± SD; null shown beside treated; deficit needs BOTH early
and delayed per-seed baselines).

## Item 1 — seed-1 low-null anomaly: REPRODUCED (not a pre-hardening artifact)

`arith_delay50…seed1_recheck` (from scratch, hardened code, --seed 1) = **0.524**,
vs the original 07-16 pre-hardening run's **0.506** (Δ +0.018). Both sit in the
low ~0.51–0.52 band — still the LOWEST of the three d50 nulls (seed2 0.626,
seed3 0.582). So seed 1's distinctive low null is a **genuine RNG draw, cleanly
reproduced**, not a silent old-code resume artifact. **The hardened-code 0.524
supersedes the pre-hardening 0.506 as the canonical seed-1 d50 null** (all other
table cells below are hardened from-scratch runs; 0.506 kept here for provenance
only).

CAVEAT — the ~0.018 gap is NOT a characterized noise floor. It is a single
(n=1) same-seed pair, and the two runs straddle the code-hardening change, so
0.018 conflates within-seed nondeterminism WITH the pre/post-hardening code
delta. It is also a different quantity from the resume/fork floor
(~0.0016–0.0023, same-weights continuation). Do NOT cite "±0.02 run noise" as a
threshold anywhere until it is replicated at ≥3 same-seed from-scratch runs
under fixed (hardened) code — the same multi-replica treatment the resume floor
received. See Item 3 note.

## Item 2 — is the delay-induced DEFICIT seed-variable? YES (driven by the delayed side)

| seed | early (delay-0) | delayed null (d50) | deficit = early − null |
|------|----------------:|-------------------:|-----------------------:|
| 1    | 0.798           | 0.524†             | 0.274 |
| 2    | 0.792           | 0.626              | 0.166 |
| 3    | 0.784           | 0.582              | 0.202 |
| mean±SD | **0.791 ± 0.007** | **0.577 ± 0.051** | **0.214 ± 0.055** |

† seed-1 null = hardened-code recheck 0.524 (canonical), not the pre-hardening
0.506. With 0.506 the row was deficit 0.292 / null 0.571 ± 0.061 / deficit
0.220 ± 0.065 — the correction shifts every figure by ≤0.018 and changes no
conclusion (see below).

The **early baseline is seed-STABLE (SD 0.007)** while the **delayed null is
seed-VARIABLE (SD 0.051)** — so the deficit's seed-variability (0.166–0.274)
comes almost entirely from the delayed endpoint, not the early one. Seed 1
remains the **deepest-hit** on both readings — lowest null (0.524 < 0.582 <
0.626) and largest deficit (0.274 > 0.202 > 0.166). **Real
secondary finding: the critical period's SEVERITY is seed-dependent** — some
seeds (seed 1) suffer a markedly deeper arith deficit from the same delay than
others (seed 2) — anchored against a tight early baseline. This is exactly why
the earlier endpoint-variance-≠-deficit-variance caveat mattered: with the
early anchor now measured, the deficit is confirmed variable rather than
inferred.

## Item 3 — cram+lr_shift (original "Q1"): LR-STARVATION confirmed at n=3

| seed | cram alone | lr_shift | cram+lr_shift | cram_lrshift − lr_shift |
|------|-----------:|---------:|--------------:|------------------------:|
| 1    | 0.574      | 0.752    | 0.746         | −0.006 |
| 2    | —          | 0.778    | 0.794         | +0.016 |
| 3    | —          | 0.772    | 0.804         | +0.032 |
| mean±SD | | **0.767 ± 0.014** | **0.781 ± 0.031** | +0.014 |

cram+lr_shift lands **far above cram alone (0.574)** — a ~0.2 lift that is
noise-proof at any plausible scale — and ≈ lr_shift alone (Δ mean +0.014, per-seed
−0.006 / +0.016 / +0.032, straddling zero with no consistent sign — a near-zero
mean with DISAGREEING signs is genuine absence of effect, not a small consistent
effect hidden by pooling, which is why the per-seed signs are reported instead of
the pooled +0.014). So adding
concentrated skill density on top of the restored early LR schedule adds no
detectable benefit over the LR restoration by itself → **cram's original failure
was LR-starvation, not the dose cap** — now n=3, not n=1. (This rests on the
robust cram-alone→cram+lr_shift jump, NOT on any "±0.02 noise floor"; that
figure is uncharacterized — see Item 1 caveat. The small +0.014 over lr_shift is
reported as sign-inconsistent, not as "within noise.") (Stats-design note stands: a d50 cram null remains uninformative about
cram-in-general; this only says "at d50, once LR is restored, density
saturates.")

## Net
Week 4 closed at n=3 across the board: lr_shift reopens (0.767 ± 0.014, all
clear the 0.718 gate), no forgetting, arith-specific (per-seed syntax control ≈
0), cram-failure mechanism = LR-starvation, seed-1 anomaly reproduced, and the
CP severity is itself seed-variable. Ready for Week 8 circuit-diagnosis
planning against a fully characterized target.
