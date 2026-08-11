# Pre-registration — 60.3M mixture replication of the attraction divergence

**Written 2026-07-27, BEFORE any of seeds 4–8 was trained or scored.** Nothing in this note may
be revised after the first new checkpoint is evaluated. Decision rules are numeric and fixed here.

Companion artifacts: `logs/sv_agreement_probe/mixture_replication.md` (the n=3 result),
`exposure_recheck.md` (the confound that was excluded), `power_60M.json` (the power basis),
`scripts/agreement_60M_power.py`.

---

## 1. What is being tested, and why only 60.3M

The attraction × scale dissociation is established in the syntax-enriched mixture
(−0.316/decade, n = 72, p = 2.3e-17) and was checked against the arithmetic mixture at n = 3/size.
That check **did not fail uniformly**. The mixtures agree at 11.5M (+0.018, p = 0.80) and 23.2M
(−0.054, p = 0.10); the entire disagreement is one cell — **60.3M, +0.334 (arith) vs +0.201
(syntax), diff +0.133, p = 0.098, n = 3**.

The leading substantive alternative (English-token exposure) was tested by re-analysis and
**excluded on four grounds** — most decisively, English exposure is held constant across sizes by
the design, so it can only produce a uniform mixture offset and predicts an interaction of
exactly zero; and the arithmetic mixture's held-out English perplexity handicap *vanishes* by
60.3M (−0.12, p = 0.61), where the divergence peaks. Full readout: `exposure_recheck.md`.

So the divergence is now a bare, single-cell anomaly with exactly two live candidates:
**(a) sampling noise at n = 3** (realized power for the observed gap was 0.21), and **(b) a
genuine mixture × scale interaction** (pooled p = 0.050). No further re-analysis can separate
them. Seeds can.

**Design choice.** Because the anomaly is localised, seeds at 60.3M dominate seeds spread across
sizes: +5 arithmetic seeds at 60.3M (n = 8) reaches **0.97** power on the observed +0.133, versus
**0.57** for the originally-proposed +1 seed at every size — same 3-to-5 run budget, on the
question that is actually open.

## 2. Runs launched

Five arithmetic-mixture d0 baselines at 60.3M, seeds 4–8:

```
arith_delay0_exp20_size60M_seed{4,5,6,7,8}
```

Config **identical to seeds 1–3** (verified against `logs/arith_delay0_exp20_size60M_seed1/
config.json`): `--skill arith --delay-onset 0.0 --skill-exposure-frac 0.20 --model-size 60M
--total-steps 5000 --checkpoint-interval 1250 --run-name sanity_babylm`, same final `vocab.bin`
/ `train.bin`, seeding is `--seed`-only. Only the seed differs. ~46 min/run, ~3.8 h total,
~14 GB transient disk. Launcher: `scripts/agreement_60M_seeds_batch.sh` (idempotent; skips any
run whose `step_0005000.pt` already exists).

## 3. Scoring — fixed now

```
python scripts/agreement_attraction_quartet.py \
    --runs arith_delay0_exp20_size{11M,23M}_seed{1,2,3} \
           arith_delay0_exp20_size60M_seed{1..8} \
    --n 400 --step 5000 --skip-sentence --tag quartet_scale_arithmix_n8
```

`build(n, verbs)` uses a fixed internal seed of 0, so `--n 400` regenerates the **identical 400
sextets** used for seeds 1–3. The syntax-mixture side (n = 24/size) is **not re-run** — it is read
unchanged from `quartet_scale.json`.

**Built-in validity checks (must pass before the primary test is read):**
- Re-scoring 60.3M seeds 1–3 must reproduce `quartet_scale_arithmix.json` **exactly**. Any
  discrepancy halts the analysis — it means the item set or checkpoint set moved.
- Unigram control attraction must be exactly **+0.0000** (analytic guarantee of the crossed design).
- Attraction > 0 in all 8 new models, both number directions (finding #1; expected trivially).

## 4. PRIMARY TEST — one test, fixed

Welch two-sample *t*, **two-sided, α = 0.05**, on `verb.attraction` at 60.3M:
**arithmetic mixture (n = 8, seeds 1–8) vs syntax mixture (n = 24, seeds 1–24).**

No other test gates the conclusion. One-sided tests, seed exclusion, metric substitution, and
changes to `--n` are **pre-banned**.

## 5. DECISION RULES

**(A) p < 0.05, arith > syntax → THE DIVERGENCE IS REAL.**
The 60.3M cross-mixture gap survives at n = 8. Finding #2 becomes: *attraction shrinks with scale
in the syntax-enriched mixture, and the arithmetic mixture departs from that trend at 60.3M* — a
genuine mixture × scale interaction, reported as an established two-mixture result with the
divergence as its main qualification. The scale claim stays mixture-conditional.

**(B) p ≥ 0.05 → THE DIVERGENCE CLOSES AS NOISE.**
*Pre-registered by the user, on the achieved power below.* The n = 3 excursion is recorded as
sampling variation. Finding #2 is relabelled to state that the scale trend **replicates across
both mixtures at all three sizes**, with the 60.3M cell now at n = 8. The mixture caveat in §6
downgrades from "the headline rests on a single mixture" to "replicated in a second mixture at
n = 3/3/8, with a resolved 60.3M excursion."

**(C) p < 0.05, arith < syntax** — not anticipated; would mean the n = 3 sign was wrong. Report as
a third outcome, do not fold into (A) or (B).

### The numeric bound that rule (B) carries — non-negotiable phrasing

Achieved power at n = 8 vs n = 24 (MC on realized SDs: arith 0.0815, syntax 0.0558):

| true effect | power |
|--:|--:|
| +0.133 (the observed gap) | **0.96** |
| +0.110 | 0.88 |
| +0.100 | 0.82 |
| **+0.098 (MDE at 80%)** | **0.80** |
| +0.090 | 0.74 |
| +0.080 | 0.64 |

So a null **closes the divergence at the magnitude that was actually observed** and excludes
effects down to ≈ **0.10**. It does **not** show the mixtures are identical, and it does not
exclude a real interaction smaller than ~0.10. Per the project's standing convention that every
underpowered verdict gets a number, rule (B) is written up as:

> "The 60.3M divergence does not survive replication at n = 8 (power 0.96 for the originally
> observed +0.133); effects below ≈0.10 remain untested."

Never as "the mixtures agree" or "no interaction exists."

## 6. Secondary readouts — reported, never decisive

These inherit n = 3 at 11.5M and 23.2M and so cannot gate anything:
- arithmetic-mixture attraction slope across the three sizes, with 60.3M at n = 8;
- mixture × log₁₀(params) interaction on the pooled set (now 86 models);
- complexity slope in the arithmetic mixture (the dissociation partner — flat is expected);
- controlled mismatch accuracy at 60.3M (the "below chance at the smallest size only" claim).

## 7. Standing rules applied

- **Control-arm seed depth.** The comparison arm (syntax, n = 24) is deeper than the arm being
  replicated, not shallower — the failure mode that rule guards against does not arise here.
- **Absolute endpoints, both sides, never a collapsed ratio.** Report
  "arith 60.3M = X ± SD (n = 8)" *beside* "syntax 60.3M = +0.201 ± 0.056 (n = 24)". No
  percent-of-gap figures, no single ratio standing in for two numbers.
- **Power from the arm being measured.** The n = 3 attempt failed because its power estimate used
  the *source* arm's SD. All numbers in §5 use the arithmetic mixture's own realized SD (0.0815).
