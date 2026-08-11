# Draft — Appendices (draft 1, 2026-08-07)

**Why this file exists.** BabyLM 2026 caps papers at eight pages and defers formatting to
EMNLP 2026 workshop standards, i.e. ACL's: *"The conference encourages submission of
appendices and supplementary material, which are not required to fit within these page
limits"*, and *"The appendices come after the references."* References are unlimited too.
So an appendix is free space — but **reviewers are not obliged to read it**, which fixes the
rule this file follows:

> Nothing an appendix carries may be load-bearing for a body claim. What moves here is
> *evidence for a methodological decision already stated and justified in the body*, never
> the statement itself. Both moves below leave a self-sufficient sentence behind, with a
> pointer.

Two moves, both out of §5:

| moved | body keeps | body loses |
|:--|:--|:--|
| A — the pruning pre-screen | pruning named as the third non-discriminating probe, with the reason (its sign is hyperparameter-set) | the λ-sweep table and the near-miss narrative |
| B — the classifier's provenance | no categorical verdicts are assigned; the repair label was unreachable here; set overlap ≠ causal overlap | the four-point provenance record of *why* the rule was unregistered |

---

# Appendix A. Pruning-based localization: the pre-screen that rejected it

§5.1 reports pruning as the third of three probes that cannot discriminate repair from
reorganization on our contrast. It was rejected by a pre-screen fixed in advance
(`notes/week8-pruning-feasibility-prereg.md`, 2026-07-27), before the full tool was built.
We report it because a rejected method is a result about the contrast, and because this
particular rejection came within one hyperparameter of looking like a confirmation.

**Method.** Learned binary gates over attention heads and MLPs, objective = arithmetic
answer-span cross-entropy + λ‖g‖₁, binarized at 0.5. The cross-model readout is
IoU(mask(A), mask(C)): the overlap between the early model's sufficient subnetwork and the
recovered model's.

**Four gates, fixed in advance; two failed.**

| gate | threshold | result |
|:--|:--|:--|
| 1. reproducibility | IoU(A, A′) ≥ 0.70 on two independent fits | **PASS**: 1.000, the two fits give the identical mask |
| 2. dynamic range | spread of IoU(A, C) ≥ 0.20 across seeds | **FAIL**: 0.088, all three inside one 0.10-wide band |
| 3. ordering | seed 1 (most reorganized by ablation) lowest | **FAIL**: seed 3 is lowest at 0.425; seed 1 is middle |
| 4. functional validity | mask retains ≥ 0.80 of accuracy | **PASS**: 0.80–0.94 on ~30/54 components; complement → 0.000 |

The metric is real: perfectly reproducible, and the mask genuinely is a sufficient
arithmetic subnetwork. It simply does not track circuit identity: A and C overlap about
the same amount whether or not C rebuilt.

**The sparsity sweep, run to pre-empt "you picked the wrong λ", found something worse than
insensitivity.**

| λ | IoU seed 1 / 2 / 3 | spread | *r* vs ablation transfer | mask retains A's accuracy | gate 4 |
|--:|:--|--:|--:|:--|:--|
| 0.01 | 0.609 / 0.511 / 0.523 | 0.098 | **−0.944** | 0.89 / 0.99 / 0.94 | pass |
| 0.03 | 0.442 / 0.513 / 0.425 | 0.088 | **+0.118** | 0.80 / 0.94 / 0.81 | pass |
| 0.06 | 0.333 / 0.529 / 0.472 | 0.196 | **+0.872** | 0.67 / 0.70 / 0.83 | **FAIL** |

Seed 1's overlap moves monotonically from *highest* to *middle* to *lowest* as the mask is
tightened. **The sign of the method's answer is set by a hyperparameter rather than by the
models**: at λ = 0.01 pruning says seed 1 reused the original circuit *more* than the
high-transfer cases; at λ = 0.06 it says the opposite.

And the λ that agrees with the primary readout (*r* = +0.872) is exactly the λ at which the
mask stops being a valid circuit: retention falls to 0.67 / 0.70, below the pre-registered
0.80 bar. At that sparsity, overlap differences plausibly track how much accuracy each mask
sacrificed rather than which components each model uses.

**The near-miss is the point.** Run at λ = 0.06 alone, this pilot would have reported "seed 1
lowest overlap, *r* = +0.87 with ablation, pruning independently confirms reorganization."
That is the same shape as the single-head patching result we retracted (§5.1), and wrong for
the same reason: a secondary method cited as corroboration without being shown to
discriminate across the full contrast set. It is why §5's claims are annotated single-method
by demonstration rather than supported by a count of agreeing probes.

---

# Appendix B: Provenance of the circuit-identity classifier's thresholds

§5.2 reports the transfer axis continuously and assigns no categorical verdict. This
appendix records why, since the alternative (relabelling under a corrected rule) was
available and was rejected.

Our implementation classified a triad as *repaired* when transfer ≥ 0.6 **and**
IoU of the two models' top-head sets ≥ 0.4. The second conjunct is never satisfied in this
dataset (maximum 0.20 across six triads), so the repair label was unreachable. Four facts
about that rule, checked 2026-08-05:

1. **It was not pre-registered.** The genuine pre-registration
   (`notes/week8-diagnosis-seed-scope.md`, 2026-07-21, written before the tooling existed)
   fixes seed scope and which diagnostics run, and specifies **no verdict thresholds and no
   overlap criterion**.
2. **It is a code-level default** introduced with the tool
   (`scripts/week8_circuit_identity.py:276-278`).
3. **It contradicts its own docstring, asymmetrically.** The comment directly above reads
   "Primary = transfer_ratio … Jaccard of the head sets *corroborates*", but the code makes
   overlap a hard conjunct on the repair branch only; the reorganization branch never
   consults it. Only one of the two verdicts could be blocked by the measure designated
   corroborative.
4. **It was contradicted by the first triad that tested it, and then formally demoted
   without the code being changed.** `notes/week8-60M-seed2-prereg.md` (2026-07-26) records
   "Jaccard is corroborative only (it did NOT discriminate at 23M: seed 2 had Jaccard 0.09
   *with* transfer 0.71)."

So "no triad was classified as repair" describes a stale conjunct, not the models.

**Why we did not simply relabel.** Dropping the overlap conjunct would move the
highest-transfer triad (23M seed 3, transfer 0.868) into a *repair* category, which is
choosing a decision rule after seeing the number it decides. Reporting the continuous axis
assumes neither rule, and the verdict column stays in the body's Table 4 so that the absence
of a repair row is visible rather than quietly dropped. The measurements are untouched
throughout; only the labelling was ever in question.

**The generalizable lesson** is in the fourth point: a decision recorded in a
pre-registration does not propagate itself into code. The demotion of overlap to
corroborative was written down, correctly, ten days before this analysis was read, and the
classifier went on applying the superseded rule because nobody grepped the implementation in
the same sitting.

---

# Appendix C: Material moved from the body in the condensed version

Appendices A and B are moves both versions make. **This appendix is different: it exists
because the condensed build moves material out of §2, §6 and §8 that the full version
carries inline.** Nothing here is a new result, and nothing here is load-bearing for a
claim: each item is evidence for a decision that is stated and justified in the body,
with the body keeping a self-sufficient version and a pointer.

## C.1 Architecture and optimizer settings

Context length is 512 tokens. Positional information is rotary (θ = 10,000); normalization
is pre-norm; activations are GELU. Attention is `F.scaled_dot_product_attention`, which
dispatches to FlashAttention-2 kernels on our GPU. Training uses bf16 autocast,
micro-batches of 8 × 512 tokens with 4 gradient accumulation steps, and AdamW
(β = 0.9/0.95, weight decay 0.1, gradient clipping 1.0). `torch.compile` is available in
the trainer but was **disabled for every run reported here**.

Two settings named in §2.1 are deliberately *not* moved here, because a reader has to be
able to check an argument with them in hand: the **cosine-decayed learning rate**
(peak 6e-4, 300 warmup steps, decay to 0.1 × peak), since §4's mechanism result is a
property of that schedule; and **untied input embeddings and output projection**, since
the reset intervention's validity rests on the two being separable.

Checkpoint spacing varies by run family and sets only how finely a trajectory can be
re-read afterwards; delay and intervention triggers fire on exact step count, never on a
checkpoint boundary:

| run family | checkpoint interval |
|:--|--:|
| 23M arithmetic | 100 steps |
| 11M and 60M | 1,250 steps |
| size sweep | 2,500 steps |

Every run reported here is 5,000 optimizer steps over 81.92M tokens.

## C.2 Parameter counts and the embedding share

Recomputed from the step-5000 checkpoint state dicts rather than from design documents
(`scripts/param_counts.py`); the script asserts that each checkpoint's stored
`model_config` matches its weights.

| tag | layers | heads | width | vocab | total | non-embedding | embed+head | embed share |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 11M | 4 | 8 | 256 | 16,384 | 11,536,640 | 3,148,032 | 8,388,608 | 72.7 % |
| 23M | 6 | 8 | 384 | 16,384 | 23,204,736 | 10,621,824 | 12,582,912 | 54.2 % |
| 60M | 8 | 10 | 640 | 16,384 | 60,304,000 | 39,332,480 | 20,971,520 | 34.8 % |

Because embeddings are untied, the two columns tell different stories: the three
configurations span **5.23 × in total parameters but 12.49 × in the parameters that do
computation**. Size comparisons in this paper are stated in total parameters, which is the
axis the pre-registered analyses used; readers translating our scale results into
non-embedding terms should note that the effective range is wider than the labels suggest,
and that the smallest model is unusually embedding-dominated.

## C.3 Why the three pre-registered analyses are not estimable

§2.6 states that all three pre-registered designs were retired and names their
replacements. They fail for three different reasons:

1. **Mixed-effects `accuracy ~ delay × model_size`.** After the pre-registered floor
   exclusion the design has twelve observations, one per seed × cell, and therefore no
   within-cell replication for a random effect to be estimated from.
2. **`recovery% ~ intervention × skill × delay_onset`.** The response divides by the
   delayed null, which is seed-variable, so the denominator moves between seeds; two of
   the three factors are constants in the surviving design, and only one intervention arm
   reaches *n* = 3.
3. **`circuit_overlap ~ intervention × skill`.** Both factors are constants.

## C.4 The attraction trajectory: power

§6 reports that attraction may also decline *within* a single run and explicitly does not
claim it. The supporting numbers: between steps 2500 and 5000 the isolated attraction cost
falls +0.392 → +0.303 (paired, *n* = 6 at 23.2M, 5/6 seeds), marginal at *p* = 0.053, with
**53 % power against this effect size and an MDE of 0.125**: roughly ten seeds would
decide it. The mismatch cell itself improves significantly over the same interval
(+0.070, *p* = 0.041), and that is the non-marginal part of the result.

Per-seed trajectories are not tabulated here because the step-2500 evaluations were not
retained per seed; only the paired aggregate was recorded.

## C.5 Seed counts for the outstanding tests

§8 names the rate-readout collapse test as the cheapest outstanding experiment in the
project. Counts assume delay *eliminates* the high-acquisition mode:

| test | seeds needed | status |
|:--|--:|:--|
| 60M @ 0.5593 | **14** | no delayed 60M seeds exist yet |
| 60M @ 0.55 (permissive) | 6 | — |
| 23M @ 0.5593 | 82 | out of reach |
| 23M @ 0.55 (permissive) | 18 | current *n* = 3 gives *p* = 0.786 |

If delay merely *halves* the mode rather than eliminating it, the requirement rises to 59
per group at 60M and 203 at 23M. The direction that matters is that feasibility **improves
with scale** (the high mode is roughly four times more common at 60M) which is why the
test is named at 60M and not at the size where the delayed contrast was actually run.

---

## Notes

- **Both appendices are optional reading by construction.** §5.1 keeps pruning's rejection
  and its reason; §5.2 keeps the unreachable-label fact and the set-overlap-versus-causal-
  overlap point, which is the one genuinely load-bearing idea in this material (it is the
  affirmative case for reciprocal ablation, and it is *not* moved here).
- **Not moved, deliberately:** the retracted single-head patching analysis stays in §5.1's
  body list. It is the reason the "single-method" annotation exists and is the paper's own
  correction record — an appendix is the wrong place for a retraction.
- Numbers owned by `logs/week8_pruning_feasibility/results.md` (+ `results_lam*.json`) and
  `notes/week9-stats-backbone.md` §2.4a.
