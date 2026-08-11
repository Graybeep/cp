# Draft — §3 Detect and §4 Reopen (draft 1, 2026-08-05)

Structure per `paper-outline.md` v2. Numbers taken from their §6 owners
(`logs/week9_stats/backbone_stats.md`, `logs/week5-7_arith_scale/gate_result.json`,
`logs/week4_close/results.md`, `logs/week4_arith_pilot/seeds23_replication_results.md`)
and, where a log row mixes lineages, from the run's own `metrics.csv` (noted inline).
Sentence→claim audits follow each section.

Budget: §3 ≈ 2.00 pages, §4 ≈ 1.75 pages, ACL two-column.

---

# §3 Detect

## 3.1 Delayed arithmetic exposure produces a persistent endpoint deficit

Every model in this section sees the same number of arithmetic tokens; the conditions
differ only in *when* those tokens enter the stream. The delayed condition (d50) begins
arithmetic exposure at the midpoint of training, the early condition (d0) at step 0, and
both are read at step 5000 on the same 500 held-out problems. Because seeds are crossed (the same seed gives the same initialization in both conditions) the design-appropriate
statistic is the per-seed paired deficit, tested separately at each size and Holm-corrected
across the two confirmatory sizes.

**Table 1.** Arithmetic exact-match accuracy at step 5000, per seed and pooled.

| size | seed | early (d0) | delayed (d50) | deficit |
|:--|--:|--:|--:|--:|
| 23M | 1 | 0.798 | 0.524 | 0.274 |
| | 2 | 0.792 | 0.626 | 0.166 |
| | 3 | 0.784 | 0.582 | 0.202 |
| | **mean ± SD** | **0.791 ± 0.007** | **0.577 ± 0.051** | **0.214 ± 0.055** |
| 60M | 1 | 0.798 | 0.574 | 0.224 |
| | 2 | 0.840 | 0.646 | 0.194 |
| | 3 | 0.848 | 0.580 | 0.268 |
| | **mean ± SD** | **0.829 ± 0.027** | **0.600 ± 0.040** | **0.229 ± 0.037** |

The deficit is present at every seed at both sizes: paired *t*(2) = 6.74, *p* = 0.0107 at
23M and *t*(2) = 10.64, *p* = 0.0044 at 60M, both surviving Holm correction within the
Detect family (adjusted 0.0107 and 0.0087; Hedges *g* = 2.22 and 3.51). The pre-registered
Welch gate, which required a gap of at least 0.15, is cleared at both sizes (60M one-sided
*p* = 0.0010). A model that receives its arithmetic exposure late does not catch up by the
end of training, despite having received all of it.

The deficit is of **similar magnitude at both sizes**: the 60M–23M difference is 0.015,
Welch *p* = 0.72, with a minimum detectable effect at 80 % power of **0.144**. That null
constrains (it excludes any size effect larger than about two-thirds of the deficit itself), but it is not a claim that size is irrelevant, and it should not be read as one. We
return to this in §5, where a structurally identical size comparison carries the opposite
evidential weight for one reason only: its MDE exceeds the entire range its outcome
occupies. The two nulls are best read as a pair.

**The severity of the deficit is seed-variable, and the variability lives on the delayed
side.** The early anchor is tight across seeds (SD 0.007 at 23M) while the delayed
endpoint is not (SD 0.051), so per-seed deficits range from 0.166 to 0.274 at 23M against a
baseline that barely moves. Seed 1 is the deepest-hit on both readings (lowest delayed
endpoint and largest deficit), and reproduces as such when retrained from scratch. This is
why every recovery result in §4 is reported as an absolute endpoint beside its null rather
than as a percentage of the gap: the gap is a seed-variable denominator, so a
recovery-percentage is unstable by construction even when the underlying effect is not.

**The 11M models are excluded by a pre-registered floor.** Their paired deficit is
nominally significant (0.181 ± 0.087, *p* = 0.0345, *g* = 1.19), and we report it because a
reader recomputing our numbers will find it. It is excluded because the 11M early anchor
(0.538 ± 0.018) never plateaued and sits below the pre-registered 0.60 floor: the
comparison is between two under-trained models, so the difference cannot be attributed to
timing rather than to capacity or budget. This is a capacity boundary, not evidence that no
critical period exists at 11M.

Intermediate onsets (d25, d55, d75) were run at 23M seed 1 only. They are plotted for
completeness without a fitted trend; at *n* = 1 per onset they describe this run and support
no claim about the shape of the timing curve.

## 3.2 Long-distance *wh*-licensing shows no detectable critical period

The same design applied to long-distance *wh*-licensing yields nothing. The five probes
below are **five different hypotheses about where an effect could hide, not five tests of
one hypothesis**; each carries its own positive control or power statement, and the argument
is that pairing rather than the count.

1. **Endpoint by delay** (*n* = 3): no gap, against a delay-0 baseline that itself ranges
   0.520–0.586 over eight seeds (a spread exceeding any candidate effect) on the same
   harness that resolves arithmetic's 0.214.
2. **Acquisition dynamics**: Welch *p* = 0.46, underpowered alone and carried by probe 3.
3. **Rate / bimodality**: a pre-registered compound gate over **72 runs** (24 seeds × 3
   sizes) returns DISCONFIRM: the mean rises monotonically with size (0.528 → 0.538 →
   0.552) but fails its floor at threshold 0.56 (top size 0.208 < 0.40) and its trend at
   0.57 (*p* = 0.074), passing only at the permissive 0.55, and this powered test, sized in
   advance by simulation, is why long-distance syntax left the main grid.
4. **Representational probe**: delay-0 versus delay-50 *p* = 0.48 *while the same probe
   detects the structure at selectivity 0.60, p < 1e-8*: the positive control passes.
5. **Learning-rate budget** (hilr50, *n* = 3): ΔLD = +0.011 ± 0.005, with Δlocal exceeding
   ΔLD at every seed and validation perplexity improving: a non-specific whole-model lift,
   not a rescue of the delayed skill.

Probe 4 is the load-bearing one, for a reason that the other four cannot supply. Probes 1,
2, 3 and 5 all ask whether a critical period is visible *in the behaviour that was
acquired*. Probe 4 asks whether the underlying structure is encoded at all, and finds that
it is, and that its encoding does not depend on when exposure began. **The structure is
learned; what is timing-invariant is the readout.**

This null is scoped to ≤ 60.3M parameters, the ~10M-word BabyLM budget, and the
continuation-scored licensing-interaction metric. Outside that scope it says nothing.

## 3.3 Skill dissociation

Under one design, one pipeline and matched total exposure, delayed exposure costs
arithmetic 0.214 ± 0.055 and costs long-distance *wh*-licensing nothing measurable across
five probes. That contrast is the paper's central detection-side result, and it inverts our
pre-registered prediction that syntax would show the *stronger* critical period: a
prediction we restate with its outcome as a standalone item in §7.2, since a reader holding
our pre-registration will look for it there rather than here.

**It must be stated in absolute terms, and not as differential susceptibility, because the
two skills are not equally observable.** The asymmetry is large enough to state as a table
rather than to note as a caveat:

**Table 2.** Why the dissociation is scoped.

| | arithmetic | syntax LD |
|:--|--:|--:|
| early-exposure anchor | 0.791 ± 0.007 | 0.528 (24 seeds) |
| chance floor | ~0 (exact match) | 0.500 (by metric construction) |
| **headroom above floor** | **0.791** | **0.0283** |
| observed deficit | 0.214 (27.1 % of headroom) | none detected |
| a proportionally matched deficit would be | — | 0.0077 (0.49 seed-SD) |
| MDE at 80 % power, *n* = 3/group | — | **0.0482** |

The long-distance design's MDE **exceeds its entire headroom by 1.7×**. A total collapse of
the skill to chance would be a change of 0.0283 (smaller than the smallest effect the
design could resolve), and a susceptibility proportionally identical to arithmetic's would
require *n* ≈ 67 per group. The behavioural null therefore cannot, on its own, separate
*syntax is less susceptible* from *syntax had no room to show susceptibility*.

Two things follow, and only one of them is a rescue. The learning-rate-budget null (probe
5) does **not** relieve the problem: it establishes that the ceiling is not an
optimization-budget artifact, which makes the ceiling real and immovable by that lever, and
so *preserves* the dynamic-range limit rather than lifting it. The representational probe
(probe 4) does: it is the one readout not bounded by the 0.0283 headroom, it carries
genuine dynamic range, and it is flat across timing. Had delayed exposure damaged the
underlying representation, that probe had the range to register it and did not.

The differential-susceptibility reading remains open, and its cost is specific rather than
unknown. Under the mean-endpoint readout it is unmeasurable at the *n* achieved here. Under
a rate readout it is **un-run**: the exceedance machinery was only ever pointed at the
delay-0 question, and the seed inventory is asymmetric (24 delay-0 seeds per size, 3 delayed
seeds at 23M alone). A rate-collapse test at 60M (14 delayed seeds against the existing
24-seed baseline) would settle it; at 23M the same test needs 82 and is out of reach (§8).

Finally, a note on what this result is not. Our pre-registration anticipated a
*reopening*-side dissociation: some skill/intervention combinations recover and others do
not. What the data support is a *detection*-side one: syntax never showed a deficit, so no
syntax reopening was attempted. The two are superficially the same finding and are not the
same claim, and we report the one we have.

### §3 sentence → claim audit

| passage | claim | label | obligation honoured |
|:--|:--|:--|:--|
| Table 1 + paired tests + Holm | **1** | ESTABLISHED | per-seed values printed with the table |
| "similar magnitude at both sizes", MDE 0.144 | **2** | ESTABLISHED | MDE inline; banned "size does not matter" avoided; §5 pair opened |
| seed-variable severity, delayed side | **3** | ESTABLISHED | licenses the absolute-endpoint rule used in §4 |
| 11M paragraph | — | NOT-A-CLAIM | number → exclusion → reason, in that order; floor wording verbatim |
| d25/d55/d75 | — | NOT-A-CLAIM | n = 1, no fitted trend |
| five probes, each with control/power | **11** | ESTABLISHED (null, scoped) | "five different hypotheses", never "confirmed five ways"; scope stated |
| probe 4 positive control; readout-vs-structure | **12** | ESTABLISHED | the two-jobs distinction un-blurred |
| dissociation + Table 2 + MDE 1.7× headroom | **11b** | ESTABLISHED (scoped) | scope in place, not deferred |
| probe 5 does not relieve; probe 4 does | **11b** | — | the natural-but-wrong inference blocked explicitly |
| rate readout un-run; 14 seeds at 60M | **11b** | — | priced, not "we cannot know"; forward ref to §8 |
| reopening-side vs detection-side | — | — | backbone §10 case (c) |

---

# §4 Reopen

## 4.1 An LR-schedule shift restores the delayed endpoint

All intervention arms fork from a single saved checkpoint at step 2500 (identical weights,
identical optimizer state, identical step), so that arms differ only in the intervention
applied. The `lr_shift` arm restores the learning-rate schedule that a delay-0 model would
have been on at the point delayed exposure begins, then anneals normally. Success was
pre-registered as a two-part condition: the treated endpoint must reach 90 % of that run's
*own* early-exposure anchor, and general capability must not degrade beyond a matched
threshold.

**Table 3.** Endpoints at step 5000. The null is shown beside the treated arm throughout;
neither is collapsed into a ratio.

| size | treated (`lr_shift`) | delayed null | early anchor | gate (0.90 × own d0) | margin | lift, *p* |
|:--|--:|--:|--:|--:|--:|--:|
| 23M | **0.767 ± 0.014** | 0.577 ± 0.051 | 0.791 ± 0.007 | 0.712 | +0.055 | 0.0065 |
| 60M | **0.796 ± 0.022** | 0.600 ± 0.040 | 0.829 ± 0.027 | 0.746 | +0.050 | 0.0027 |

Per-seed treated endpoints are 0.752 / 0.778 / 0.772 at 23M and 0.796 / 0.818 / 0.774 at
60M. Every seed at both sizes clears its gate, and the per-seed lift over the matched null
survives Holm correction within the Reopen family (adjusted 0.0065 and 0.0054).

Two features of this table are more surprising than the recovery itself.

**The recovered endpoint is scale-invariant.** Both treated means fall inside the
pre-registered same-endpoint band [0.722, 0.812], despite early anchors that differ by
0.038 and a 2.6× difference in parameters. Recovery converges on a level rather than
tracking capacity.

**The intervention collapses cross-seed variance.** The null's SD (0.051 at 23M, 0.040 at
60M) drops to the treated arm's (0.014, 0.022). The seed-variable severity documented in
§3.1 is a property of the *untreated* delayed models; the intervention makes the outcome
both higher and more reproducible.

## 4.2 The recovery does not come at the cost of general capability

Perplexity is reported as the three-number decomposition (early anchor *a*, delayed null
*b*, treated arm *c*), because a single delta cannot distinguish a treated arm that
improved from a null that degraded.

At **23M there is no breach**: *c* − *b* = −0.889 / −0.204 / +0.193 across seeds, against a
pre-registered gate of +1.0 (seed 1: *b* = 50.747, *c* = 49.858, both from the canonical
`…seed1_recheck` lineage). Two of three seeds improve.

At **60M the gate is tripped in all three seeds**: *c* − *b* = +1.670 ± 0.104 ppl. We report
it as a trip and honour it as written. Two controls bound what it can mean. A no-op fork
reproduces the null on both perplexity and accuracy (Δppl −0.012, Δacc +0.011), so fork
mechanics contribute nothing. And at **matched learning rate** the treated arm *leads* the
null by 2.478 ± 0.240 ppl (a lead larger than the breach), because the arm is read
mid-anneal at roughly six times the null's terminal learning rate, with about 4.15 ppl of
annealing gain still unrealized. The breach persists only if the arm's remaining anneal
delivers less than 1.67 ppl of that. The evidence leans toward near-full erasure, but the
anneal slope is unmeasured, so we quote **no point estimate**: the residual cost is bounded
above by 1.67 ppl and a partly real cost is not excluded. Extending the arm past step 5000
is the run that would collapse this bound in either direction (§8).

## 4.3 The effect is specific to the delayed skill

The same intervention applied to delayed *syntax* models (replicated at all three seeds,
matching the depth of the arithmetic arm) moves nothing: Δ local = −0.024 / −0.004 /
+0.044 and Δ LD = −0.021 / +0.007 / +0.015, sign-inconsistent around zero at every seed.
A single-seed control could not have distinguished "no generic effect" from "an effect that
happened to land near zero once", which is why the control was run at the arm's own seed
depth. Whatever `lr_shift` does, it is not a generic late-training lift.

## 4.4 Closure at this onset is largely a property of the schedule

The remaining Tier-1 interventions were run at 23M seed 1 and are scored against the null's
cross-seed **range** [0.524, 0.626] rather than its mean, because a single seed landing
inside that range is indistinguishable from the null.

| arm (23M, seed 1) | endpoint | verdict |
|:--|--:|:--|
| embedding reset | 0.586 | no detectable recovery (*n* = 1) |
| reset + LR spike | 0.576 | no detectable recovery (*n* = 1) |
| LR spike alone | 0.536 | no detectable recovery (*n* = 1) |
| cram (concentrated re-exposure) | 0.574 | no detectable recovery (*n* = 1) |
| **freeze** | **0.332** | **harmful**: below the entire null range |

Only freeze supports a directional statement at this depth. The others are *not* evidence
that those interventions fail; they are evidence that one seed cannot separate them from a
null whose own spread is 0.10 wide. Embedding reset was our pre-registered favourite, and
§7.2 states that prediction and its outcome as a standalone item; here it is one row of a
table, at *n* = 1.

The one arm we extended to *n* = 3 decides the mechanism. Concentrated re-exposure alone
leaves the endpoint at 0.574: at the null. Combined with the schedule shift it reaches
0.781 ± 0.031, far above cram alone and indistinguishable from the shift by itself: the
per-seed differences are −0.006 / +0.016 / +0.032, straddling zero with no consistent sign.
(We report the signs rather than the +0.014 mean, because a near-zero mean with disagreeing
signs is an absence of effect, not a small effect obscured by pooling.)

So additional skill density on top of a restored schedule adds nothing, while a restored
schedule without additional density recovers the skill. **At this onset, what closes the
window is largely the optimizer's decayed learning-rate schedule rather than representations
that have hardened against revision.** The scope of that statement is d50, arithmetic, and
these two sizes; it is not a claim that entrenchment plays no role at other onsets.

A final arm belongs in the record without supporting a claim. The 11M `lr_shift` arm passes
its gate (0.747 against a gate of 0.484) but the pass is degenerate: the gate is 90 % of an
anchor that failed the §3.1 floor, and the arm *exceeds its own early anchor* (0.538),
which indicates that the 11M anchor was learning-rate-starved rather than that reopening
over-recovers. The 11M arm enters the paper only through its mechanism decomposition, where
the delayed lift of +0.389 splits into a generic component (+0.115) and an
exposure-alignment component (+0.274): about 70 % of the effect requiring the delayed
exposure profile. That decomposition is exploratory and carries no headline.

### §4 sentence → claim audit

| passage | claim | label | obligation honoured |
|:--|:--|:--|:--|
| common-checkpoint fork, gate definition | — | — | comparability precondition stated before any endpoint |
| Table 3 + per-seed + Holm | **4** | ESTABLISHED | endpoints paired with nulls; no percent-of-gap anywhere |
| band [0.722, 0.812] | **5** | ESTABLISHED | foregrounded in the lede, per backbone §12 |
| SD 0.051→0.014, 0.040→0.022 | **6** | ESTABLISHED | ties back to §3.1's seed-variability |
| 23M three-number ppl, canonical lineage | **10** | — | §5 rule 2 lineage; deviation from the log table flagged |
| 60M breach + two controls + no point estimate | **10** | BOUNDED-NOT-EXCLUDED | quoted with its bound; forgetting pitfall discharged here |
| syntax control at n = 3 | — | — | control-arm seed-depth rule |
| n = 1 table scored against the null's range | **9** | SUGGESTIVE (n = 1) | "no detectable recovery", never "reset fails"; freeze the only directional read |
| cram vs cram+shift, per-seed signs | **7** | ESTABLISHED | signs over pooled mean; mechanism as consequence, scoped |
| 11M degenerate gate + H2/H3 split | **8** | NOT-A-CLAIM / SUGGESTIVE (exploratory) | degeneracy explained, not just reported |

---

## Drafting notes for the next pass

- **§5 must open with the back-reference** to §3.1's MDE pair. This is now enforceable
  rather than remembered: backbone §5.5 makes it a *paired-null dependency* between claim 2
  and the C4 size null, flagged on both ledger endpoints (claim 2's row and the §8
  scale-as-driver entry). Neither may ship without the other.
- **Prediction cross-references are now in the prose** (§3.3 for #1, §4.4 for #2), each
  naming §7.2 and saying why the restatement exists. The redundancy is deliberate and must
  survive copy-editing.
- **Figure slots — BUILT 2026-08-06, `scripts/make_figures.py` → `logs/figures/`.** Fig 1 is
  §3.1's per-seed paired plot with 11M greyed and the dose points in their own panel, no
  trend line; Fig 2 is §4's endpoints beside their nulls with the pre-registered band drawn.
  Both are charged at 0.30 page each in `logs/length_audit.md`.
- **Length: the five-probe compression is spent, and it bought almost nothing.** Applied
  2026-08-07 (each probe now one sentence): §3.2 went 347 → 343 words, §3 as a whole
  1,550 → 1,548. The list was already dense — its length is numbers and qualifiers, which
  is exactly what may not be dropped. §3 remains **+0.31 page** over budget
  (`logs/length_audit.md`). What is left in §3 is 3.3's prose, now that Table 2 carries the
  argument. The probe-4 paragraph and the headroom table stay whatever happens.
