# Pre-registration — 60M syntax rate-collapse test

**Written 2026-08-11, before any delayed 60M syntax run existed.** Batch launched
23:14 the same evening (`scripts/rate_collapse_60M_batch.sh`, driver pid 866).
Nothing in this file was chosen after seeing a delayed 60M number.

This is the experiment §8 of the manuscript names as "the cheapest outstanding
experiment in the project and the only one that can promote the dissociation past
its current scope."

---

## 1. The question

The paper's central detection-side result is a dissociation: delayed exposure costs
arithmetic 0.214 and costs long-distance *wh*-licensing nothing measurable. That is
claimed in **absolute** terms only. The stronger reading — that syntax is genuinely
*less susceptible* to delayed exposure — is currently **un-run**, not unknowable.

A reviewer's version of the objection: *you have not shown a dissociation, you have
shown a power difference between two measurement designs.*

The mean-endpoint readout cannot settle it (syntax has 0.028 of headroom against a
0.048 MDE; it needs ~67 seeds/group). The **rate** readout can, because it is a
different statistic: a rate can collapse (0.25 → 0) while the mean barely moves.
That machinery — the μ_low + 3σ_low exceedance threshold and P(high) — was built for
the size sweep and only ever pointed at the delay-0 *reposition* question. The
d0-vs-d50 rate contrast at 60M has never been run.

## 2. Design

| | |
|:--|:--|
| skill | syntax, exp10, pool v1 |
| size | 60M (8L, 640d, 10h; 60.3M params) |
| delayed arm | **24 seeds**, delay onset 0.50, seeds 1–24, 5000 steps |
| baseline arm | the **existing** 24 delay-0 seeds (`syntax_delay0_exp10_size60M_seed{1..24}_szsweep`) — not re-run |
| run ids | `syntax_delay50_exp10_size60M_seed{N}_ratecollapse` |
| readout | LD *wh*-licensing interaction, continuation-scored, step-5000 final checkpoint |

Everything except delay onset is held at the size-sweep settings, so the delayed arm
is directly comparable to the cached baseline (`logs/szsweep_eval/per_seed_ld.json`).

### Why 24 seeds and not the 14 the paper priced

§8 prices this test at 14 delayed seeds. That number is correct but knife-edged: at
the derived threshold it clears α **only on an exact 0/14**.

| n | observe 0 | observe 1 | observe 2 |
|--:|--:|--:|--:|
| 14 | 0.0488 ✓ | 0.1767 ✗ | 0.3646 ✗ |
| 20 | 0.0191 ✓ | 0.0793 ✗ | 0.1875 ✗ |
| **24** | **0.0110 ✓** | **0.0486 ✓** | 0.1224 ✗ |

(one-sided Fisher exact against the d0 count 6/24 at threshold 0.5593)

At 14 seeds, a single high delayed seed turns 12 GPU-hours into an inconclusive
result. 24 seeds keep the test significant if one seed comes out high, and match the
baseline's n exactly. The extra cost is ~8 GPU-hours. **This is a power decision made
before running, not a stopping rule** — all 24 runs complete before the statistic is
computed, and there is no interim peeking at the decision statistic.

## 3. Threshold

The exceedance threshold is inherited, not chosen here: **μ_low + 3σ_low = 0.5593**,
derived from the 11M low cluster (μ_low = 0.5283, σ_low = 0.0103, n_low = 24/24).

Observed d0 exceedance counts at 60M:

| threshold | d0 count | status |
|:--|--:|:--|
| 0.55 (permissive) | 12/24 | **secondary** |
| **0.5593 (derived)** | **6/24** | **PRIMARY** |
| 0.57 (strict) | 2/24 | **excluded a priori — unpowered** |

**0.57 is excluded before seeing data, and the reason is arithmetic, not taste:** with
only 2/24 baseline seeds above it, even a perfect 0/24 delayed result yields p = 0.2447.
It cannot reach significance under any outcome, so reporting it as a "failed threshold"
would be misleading. The size-sweep gate's compound "all three thresholds" rule is
therefore **not** used here — it would be a rule that cannot pass by construction.

A run counts as *high* only under the same guard the size-sweep gate used
(`guarded_high`): LD ≥ threshold **and** both LD paradigms ≥ 0.15, so a run cannot be
scored high on a collapse to the unigram prior.

## 4. Decision rule

**Primary.** One-sided Fisher exact, delayed 24 vs baseline 24, at threshold 0.5593,
α = 0.05. Direction: delayed rate < baseline rate.

- **p < 0.05** → the high-acquisition mode collapses under delay. Syntax *is*
  susceptible on the rate readout; the dissociation's differential-susceptibility
  reading is **refuted**, and the paper's headline changes.
- **p ≥ 0.05** → see §5. This is *not* automatically support for the dissociation.

**Secondary, threshold-free** (reported always, labelled secondary): Mann-Whitney U on
the 24 vs 24 continuous LD distributions, plus the difference in means with a
bootstrap CI. This is pre-registered because the rate test is powered against only one
alternative (below), and a continuous shift is the alternative it is worst at seeing.

**Multiplicity.** Two tests in this family (Fisher at 0.5593; Mann-Whitney). Holm
correction within the family. The 0.55 threshold is a robustness check on the primary,
not an independent test, and is reported without correction and labelled as such.

## 5. What each outcome licenses — asymmetric, and stated in advance

Power at n = 24 vs 24, threshold 0.5593, α = 0.05 (4000 sims/cell):

| true delayed rate | scenario | power |
|:--|:--|--:|
| 0.000 | delay **eliminates** the high mode | **1.000** |
| 0.125 | delay **halves** it | **0.188** |
| 0.250 | no collapse | 0.009 (type-I) |

**This test is well powered against full collapse and badly powered against partial
collapse.** That asymmetry is pre-registered so it cannot be quietly ignored at
write-up:

- A **significant** result is decisive and flips a headline claim.
- A **null** result excludes *full* collapse and nothing more. It must be written as
  **BOUNDED-NOT-EXCLUDED**, with the bound stated: a halving of the high mode would be
  missed 81% of the time, and detecting it needs ~59 seeds/group. It must **not** be
  written as "syntax shows no rate collapse", and it must **not** be added to the
  existing five behavioural nulls as a sixth route — under the standing cross-method
  rule, an underpowered probe agreeing with the others is not corroboration.

The null outcome therefore *narrows* the scope annotation on the dissociation claim; it
does not remove it. The claim ledger entry (11b) keeps **ESTABLISHED (scoped)** either
way, with the scope sentence rewritten to name what this test excluded.

## 6. Pre-committed nuisance decisions

- **Failed runs.** A run that does not reach step 5000 is re-launched by the supervisor.
  If a seed cannot complete after `--max-stalls` (3), it is reported as missing and the
  test runs at the achieved n, with the realised n and its recomputed α stated. Seeds
  are never swapped for "better-behaved" ones.
- **The high-draw re-run rule.** The size-sweep pre-registration requires re-running the
  single highest seed once before any headline claim leans on it. If the delayed arm's
  top seed lands above the threshold, it is re-run once under the same rule, and both
  values are reported.
- **No seed is excluded for being an outlier.** The exceedance statistic is a count; an
  outlier is the signal, not noise.
- Analysis outputs go to `logs/rate_collapse/`, per the analysis-outputs-in-logs rule.

## 7. Relationship to the existing claim ledger

This test can modify exactly one ledger row (**11b**, the arithmetic-vs-syntax
dissociation), and only its scope sentence unless the primary is significant, in which
case the row is rewritten. It does not touch claims 1 (arith CP) or 11 (syntax null):
those rest on the mean-endpoint and representational-probe readouts, which this
experiment does not re-measure.
