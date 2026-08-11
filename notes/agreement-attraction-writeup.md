# A linear-proximity heuristic dominates subject–verb agreement in small-corpus LMs, and shrinks with scale

**Scope.** Workshop note / paper section. A secondary finding that emerged while searching for
a second critical-period-bearing skill for the main study; not a standalone paper, and scoped
accordingly. All results are evaluation-only on checkpoints trained for other purposes —
**no model was trained for this analysis.**

---

## Summary

Transformer LMs trained from scratch on a 10M-word corpus acquire simple subject–verb agreement
almost perfectly (0.98 on controlled items) but fail when a number-mismatching noun intervenes
between subject and verb. A fully-crossed item design, in which the verb-form frequency prior
cancels **exactly**, separates two components of that failure — **attraction** (the intervening
noun's *number* mismatches the head) from **complexity** (any intervening material at all).

Three results, at explicitly different evidential strength:

1. **ESTABLISHED, AND REPLICATED ACROSS TRAINING MIXTURES — attraction is real, large, and the
   dominant component.** +0.303 ± 0.064 at 23.2M (p = 8.6e-05, 6/6 models, both number
   directions, both frame types), against a complexity cost of +0.143 ± 0.035. Attraction > 0
   in **81/81 models** across two independent mixtures — 72 syntax-enriched (3 sizes × 24 seeds)
   and 9 arithmetic (3 sizes × 3 seeds, +0.347 ± 0.107, p = 1.05e-05) — and exceeds complexity
   at every size.
2. **THE SCALE TREND REPRODUCES ACROSS MIXTURES; THE DISSOCIATION FROM COMPLEXITY DOES NOT.**
   In the syntax-enriched mixture attraction falls **−0.316 per decade of parameters**
   (r = −0.803, p = 2.3e-17, n = 72): +0.430 → +0.314 → +0.201 across 11.5M / 23.2M / 60.3M,
   with complexity **flat** (+0.016/decade, p = 0.43) — a dissociation specific to the syntactic
   component. A second mixture (arithmetic) reproduces **the attraction half in direction and
   magnitude** — **−0.207/decade** (n = 14, 60.3M at n = 8), stable at −0.21 to −0.24 under every
   estimator tried — but **at borderline, estimator-dependent significance**: p = 0.033 under OLS,
   **p = 0.073 under a heteroscedasticity-robust SE**, p = 0.26 when the noisy 11.5M endpoint is
   variance-downweighted (§4.2.4). It does **not** reproduce the dissociation — complexity falls
   there too (−0.112/decade, p = 0.012 OLS / 0.055 robust), and the paired within-model contrast
   is flat (−0.095, p = 0.36, vs −0.333, p = 3.8e-11 in the syntax mixture) under *every*
   estimator, which is the one part of this comparison that is not fragile.
   Of two earlier qualifications, one is **closed** — the English-exposure confound is excluded
   (§4.2.2) — and one is **bounded, not closed**: the 60.3M cross-mixture divergence **does not
   survive replication** at n = 8 (+0.065, p = 0.091, halved from +0.133; power 0.93), with the
   pooled interaction weakening from p = 0.050 to p = 0.140 — but the 95% CI still contains
   +0.133, so that divergence is **unreplicated, not excluded**. So the surviving caveats are
   narrow and specific: **complexity's flatness is a single-mixture result**, the second mixture's
   attraction slope is significant only under the least conservative estimator, and cross-mixture
   effects below ≈0.11 remain untested. See §4.2.1–4.2.4 and §6.
3. **SUGGESTIVE ONLY — weak evidence of unlearning within training.** On the isolated measure
   attraction falls +0.392 → +0.303 between step 2500 and 5000, but at n = 6 this is
   **marginal (p = 0.053, 5/6 seeds)**. The mismatch cell itself improves significantly
   (+0.070, p = 0.041). We report the direction; we do not claim the effect.

**On "below chance."** Whether these models are *anti-syntactic* — actively preferring the
ungrammatical form — turns out to be **scale-dependent**, and the standard metric gets it wrong
at every scale. Under controlled items, mismatch accuracy is **0.405 at 11.5M (below chance,
p < 0.001)**, **0.511 at 23.2M (at chance, p = 0.435)**, and **0.610 at 60.3M (above chance,
p < 0.001)**. The uncontrolled BLiMP metric reports below-chance behaviour at *all three*
sizes. So the anti-syntactic claim is true only at the smallest scale, and the metric that
appears to establish it everywhere is inflated by item difficulty. See §5.

---

## 1. Setup

**Models.** Decoder-only transformers trained from random initialization; no pretrained
checkpoints anywhere. RoPE, pre-norm RMSNorm, GELU, **untied** input/output embeddings, BPE
vocabulary of 16,384 trained on the corpus itself, context 512.

| tag | layers | heads | d_model |
|:--|--:|--:|--:|
| 11.5M | 4 | 8 | 256 |
| 23.2M | 6 | 8 | 384 |
| 60.3M | 8 | 10 | 640 |

**Data and training.** BabyLM Challenge *strict-small* (~10M words, 1,058,740 lines).
5,000 steps ≈ 81.9M tokens; AdamW (lr 6e-4, cosine decay to 0.1×, 300 warmup steps, weight
decay 0.1, β = 0.9/0.95, grad clip 1.0), micro-batch 8 × grad-accum 4, bf16.

**Checkpoint sets.** (a) Six 23.2M models — two training mixtures (20% synthetic arithmetic;
10% syntax-enriched text) × 3 seeds — used for the detailed decomposition and the trajectory,
at steps 2500 and 5000. Both mixtures are ≥80% BabyLM and neither contains agreement-targeted
data, so agreement reflects what the corpus alone teaches; reporting both guards against the
injection driving the result (they agree to within about one seed SD). (b) 72 models —
3 sizes × 24 seeds — for the scale analysis.

---

## 2. The metric problem

Every BLiMP agreement pair differs in exactly one word — the verb form (`hasn't`/`haven't`,
`references`/`reference`). Raw accuracy is therefore dominated by the unigram prior on that
word. An explicit **frequency-only model** (sentences scored by summed unigram log-probability
from the training corpus — no context, no syntax) confirms this: it scores **0.09–0.24 on
singular-subject items and 0.81–0.91 on plural-subject items**, because bare/plural verb forms
are far more frequent than their `-s` singular counterparts.

**First fix — number-balanced interaction:** mean(accuracy on singular-subject items, accuracy
on plural-subject items). Any pure number preference scores 1.0 on one subset and 0.0 on the
other, hence exactly 0.5. The frequency-only model duly collapses to **0.488 (attractor) /
0.524 (no-attractor)** ≈ chance.

Balanced BLiMP scores, n = 6 at 23.2M:

| condition | balanced interaction |
|:--|:--|
| simple (no-attractor) agreement | 0.726 ± 0.020 |
| attractor (Linzen-style) agreement | 0.383 ± 0.042 |
| frequency-only control | 0.524 / 0.488 |

**Why that is still not enough.** Every BLiMP distractor item has the attractor number
*mismatching* the head; there is no condition where it agrees. So 0.383 conflates attraction
with "these items are longer and contain a PP or a relative clause." The singular/plural
asymmetry cannot disambiguate either, since the frequency prior pushes the same way. Hence §3.

---

## 3. The crossed design

For each (head noun, attractor noun, frame, verb) combination we generate a fully-crossed
**sextet** sharing lexical content:

```
bare_sg       The farmer is / are very happy.                    good = is
bare_pl       The farmers are / is very happy.                   good = are
sg_match      The farmer near the clerk is / are very happy.     good = is
sg_mismatch   The farmer near the clerks is / are very happy.    good = is
pl_match      The farmers near the clerks are / is very happy.   good = are
pl_mismatch   The farmers near the clerk are / is very happy.    good = are
```

Frames: four prepositional (`near/next to/beside/behind the N`) and three object-relative
(`that the N visited/liked/watched`). Relative-clause verbs are **past tense on purpose** —
past-tense verbs are number-neutral in English, so the relative clause adds no agreement cue.
Verb pairs: `is/are`, `was/were`, `has/have`, `does/do`, `isn't/aren't`, `wasn't/weren't`.
Heads and attractors include irregular plurals (`man/men`, `woman/women`, `child/children`),
possible here because number is stipulated by construction rather than inferred.

**The key property.** Within a head number, match and mismatch use the *identical* verb pair
with the *identical* correct answer; only the attractor noun's number differs. The verb-form
frequency prior cancels **exactly, not on average**, and length/structure is held constant.

**Derived quantities.** attraction = acc(match) − acc(mismatch); complexity = acc(bare) −
acc(match); both balanced over head number. Note the sign convention: these are *deficits*,
so **larger = worse**, the opposite of the BLiMP scores in §2.

**Two design checks, run before any model was scored.** (i) All six verb pairs tokenize to
equal token counts — summed log-probability over unequal counts is length-biased (fewer tokens
→ higher score), which would masquerade as a syntactic preference; pairs failing this are
dropped by construction, and none failed. (ii) The **frequency-only model returns attraction of
exactly +0.0000** (and complexity +0.0000), confirming the cancellation numerically.

**Scoring.** Conditional log-probability of the verb tokens given the identical prefix.
Whole-sentence scoring gives the same direction (+0.337) but is the *same instrument scored
differently*, not an independent method, and is not treated as corroboration.

---

## 4. Results

### 4.1 Decomposition at 23.2M (n = 6, 1,000 sextets = 12,000 sentences, 0 items dropped)

| cell | balanced accuracy |
|:--|:--|
| bare | **0.982 ± 0.014** |
| attractor **matches** head number | **0.839 ± 0.033** (vs chance p = 2.0e-06) |
| attractor **mismatches** head number | **0.535 ± 0.071** (vs chance p = 0.275) |

| component | value | statistics |
|:--|:--|:--|
| **attraction** | **+0.303 ± 0.064** | t = 11.52, **p = 8.6e-05**; positive **6/6** in both number directions (sg +0.255, pl +0.352) |
| **complexity** | **+0.143 ± 0.035** | p = 1.8e-04 |

Attraction exceeds complexity by **2.12×**, in 6/6 models (paired p = 0.0047), and appears in
both frame types — prepositional 6/6, object-relative 6/6 — with no consistent ordering
(paired difference +0.019 ± 0.240, p = 0.85). It is not an artifact of one construction.

### 4.2 Scale: the decomposition across 72 models (n = 24 seeds per size, 400 sextets)

| size | bare | match | mismatch | **attraction** | complexity |
|:--|:--|:--|:--|:--|:--|
| 11.5M | 0.969 ± 0.019 | 0.835 ± 0.045 | **0.405 ± 0.054** | **+0.430 ± 0.071** | +0.134 ± 0.043 |
| 23.2M | 0.956 ± 0.036 | 0.825 ± 0.066 | **0.511 ± 0.065** | **+0.314 ± 0.081** | +0.131 ± 0.057 |
| 60.3M | 0.956 ± 0.030 | 0.811 ± 0.065 | **0.610 ± 0.055** | **+0.201 ± 0.056** | +0.145 ± 0.055 |

- **attraction vs log₁₀(params): −0.316 per decade** (r = −0.803, p = 2.3e-17);
  11.5M → 60.3M = **−0.229**, Welch p = 5.6e-16.
- **complexity vs log₁₀(params): +0.016 per decade** (r = 0.094, **p = 0.43**) — flat;
  11.5M → 60.3M = +0.011, p = 0.44.
- attraction > 0 in **24/24 at every size** (72/72 overall).
- attraction > complexity in 24/24, 22/24, 17/24 by size (ratios 3.20× / 2.39× / 1.38×) —
  still dominant everywhere, but decreasingly so.

Within this mixture the pattern is clean: scale acts **specifically on the syntactic component**
— the proximity heuristic weakens by more than half from 11.5M to 60.3M while the cost of
intervening material does not move at all. Because complexity is flat, this also rules out the
alternative that larger models are simply better at longer or more complex sentences in general.
**The falling-attraction half of this pattern reproduces in a second mixture; the
complexity-stays-flat half — i.e. the dissociation itself — does not survive a change of mixture.
See §4.2.1–4.2.4, which is the reason this is no longer presented as the note's strongest
result.**

### 4.2.1 Second-mixture replication: the dissociation does not reproduce

All 72 models above share one training mixture (10% syntax-enriched). Because
`arith_delay0_exp20` checkpoints exist at all three sizes (seeds 1–3, step 5000), the mixture
control is evaluation-only; it was run with the same 400 sextets and scoring, against a
pre-registered read (CONFIRM = negative slope at p<0.05, monotone ordering, complexity flat).

| | syntax-mix (n = 24/size) | arith-mix (n = 3/size) |
|:--|:--|:--|
| attraction 11.5M | +0.430 ± 0.071 | +0.448 ± 0.106 |
| attraction 23.2M | +0.314 ± 0.081 | **+0.260 ± 0.036** |
| attraction 60.3M | +0.201 ± 0.056 | **+0.334 ± 0.082** |
| attraction slope | **−0.316/decade, p = 2.3e-17** | **−0.139/decade, p = 0.28** |
| complexity slope | +0.016/decade, p = 0.43 | −0.059/decade, p = 0.27 |
| 11.5M→60.3M gap | +0.229 [+0.192, +0.266] | +0.114 [−0.106, +0.334] |
| monotone 11>23>60 | yes | **no** (60.3M above 23.2M) |

**No CONFIRM criterion is met.** The slope is half-size and non-significant, the ordering is
non-monotone, and the dissociation *pattern* cannot even be evaluated because neither component
moves significantly here. What *does* replicate is attraction itself: +0.347 ± 0.107, positive
**9/9** in both number directions, p = 1.05e-05, unigram control again exactly +0.0000.

**The test is uninformative rather than refuting**, and the pre-registered power estimate for it
was wrong. It assumed the *syntax* mixture's within-size SD (0.063); the arithmetic mixture's is
**0.094, 1.48× larger**, so realized power at n = 3 for a 0.229 gap was **0.62, not the 0.90
claimed** — and only 0.21 for the gap actually observed. Reaching 80% power needs 4 seeds/size
for a 0.229 effect, 12 seeds/size for a 0.114 one. Neither exists; both would cost training.

Pooled over all 81 models, the **mixture × log₁₀(params) interaction is +0.177 ± 0.089,
p = 0.050** — right at the boundary: weak evidence the slopes genuinely differ, enough to block
the claim but not to establish an interaction. The divergence is concentrated at the top of the
range: 11.5M agrees across mixtures (+0.018, p = 0.80), 23.2M nearly so (−0.054, p = 0.10), and
**60.3M diverges** (+0.133, p = 0.098) — the syntax mixture's largest models reduce attraction,
the arithmetic mixture's do not.

Full readout: `logs/sv_agreement_probe/mixture_replication.md`.

**Superseded in part.** This section is the n = 3/size attempt. The 60.3M cell was subsequently
taken to n = 8 (§4.2.3), which changes the arithmetic-mixture slope from −0.139 (p = 0.28) to
−0.207 (p = 0.033 OLS) and the pooled interaction from p = 0.050 to p = 0.140. The
CONFIRM/DISCONFIRM read recorded above stands as the pre-registered n = 3 outcome; §4.2.3–4.2.4
carry the current numbers.

### 4.2.2 The English-exposure alternative, tested and excluded

The mixtures are **not matched on natural-language volume** (20% synthetic arithmetic vs 10%
syntax-enriched *text*), which suggested a substantive alternative to a scale effect: if
attraction reduction tracks **English exposure** rather than parameter count, the arithmetic
models — seeing less English at every size — should behave like under-trained large models, and
arith-60.3M's excess attraction would be an exposure artifact. This is re-analysis of existing
runs, so it was tested directly (`scripts/agreement_exposure_recheck.py`; full readout
`logs/sv_agreement_probe/exposure_recheck.md`). **It is excluded, on four independent grounds.**

**(i) The mapping it requires does not exist.** From each run's `realized_exposure.json`, English
totals are 81,920,000 tokens for the syntax mixture (base 73.73M + 8.19M of wh-question *English*)
and 65,536,000 for the arithmetic mixture (base only; the 16.38M arithmetic stream is synthetic).
Both mixtures train on an identical 81.92M-token budget, and **exposure is held fixed across
sizes by design — only parameters vary.** Within-mixture variance in English tokens is exactly
zero, so arith-60.3M is *equidistant* from all three syntax cells. A variable with no
within-mixture variance can produce only a **uniform offset between mixtures, never a
size-dependent divergence**: it predicts an interaction of exactly zero, and the entire anomaly
is an interaction. The only size-varying repair — English tokens *per parameter* — maps
arith-60.3M onto a syntax model of 48M–75M depending on the reading and accounting, never near
23.2M.

**(ii) The magnitude ceiling is 4.3× too small.** Under the fitted syntax slope, the largest
uniform offset any exposure variant can generate is **|0.031|** (|0.016| on the base-corpus-only
accounting behind the "~10% less English" figure). The 60.3M divergence is **+0.133**. Producing
it from exposure alone would require the arithmetic models to have 0.38× the syntax mixture's
English; they have 0.80×.

**(iii) The observed offsets are not uniform — they change sign.** +0.018, −0.054, +0.133 across
11.5M / 23.2M / 60.3M. No constant fits that shape.

**(iv) The mechanism it needs is measurably absent, at the one size that matters.** `val.bin` is
the same held-out English for every run in the project, so the English handicap is directly
observable:

| size | arith-mix ppl | syntax-mix ppl | English penalty | Welch p |
|:--|--:|--:|--:|--:|
| 11.5M | 59.75 | 57.07 | **+2.68** (+4.7%) | 4.1e-05 |
| 23.2M | 48.55 | 46.82 | +1.73 (+3.7%) | 0.22 |
| 60.3M | **41.48** | 41.60 | **−0.12 (−0.3%)** | 0.61 |

The 20%-less-English penalty **shrinks monotonically with scale and has vanished by 60.3M**,
where the arithmetic models are statistically tied with — and numerically ahead of — the
syntax models on held-out English. Correlating the per-size English penalty against the per-size
attraction divergence gives **r = −0.74**; the exposure account requires it to be *positive*.
The mixtures are best matched on English competence at exactly the size where attraction
diverges most.

**What this does and does not settle.** It removes the leading substantive confound: the 60.3M
divergence is not an exposure artifact. It does **not** establish the scale claim — the
divergence still rests on one cell at n = 3 (realized power 0.21), and sampling noise remains
fully live alongside a genuine interaction. The value of the check is that it converts a
two-hypothesis ambiguity into a single under-powered cell, which seeds can resolve and
re-analysis cannot.

### 4.2.3 The 60.3M cell replicated at n = 8: the divergence does not survive

The divergence was one cell at n = 3, and the exposure confound was excluded by re-analysis, so
the remaining candidates were sampling noise and a genuine interaction. Five further arithmetic
60.3M seeds were trained (identical config, `--seed` the only difference), against rules
pre-registered before any of them existed
(`notes/agreement-60M-mixture-replication-preregistration.md`; primary test = Welch two-sided,
arithmetic n = 8 vs syntax n = 24; null pre-registered as "closes the divergence as noise" with
the MDE attached). Both validity gates passed — re-scoring seeds 1–3 reproduced the earlier
values exactly, and the unigram control was again +0.0000.

| | mean ± SD | n |
|:--|--:|--:|
| arithmetic mixture, 60.3M | +0.266 ± 0.092 | 8 |
| syntax mixture, 60.3M | +0.201 ± 0.056 | 24 |

**diff +0.065, 95% CI [−0.013, +0.143], t = 1.90, df = 8.8, p = 0.091 → rule (B).** The gap
halved from +0.133; the five new seeds averaged +0.225 ± 0.077 against the original three at
+0.334 ± 0.082, i.e. the first three had drawn two of the eight highest values. The pooled
mixture × scale interaction weakens from p = 0.050 to **p = 0.140** (+0.109 ± 0.073, n = 86), and
no per-size contrast is significant (11.5M p = 0.80, 23.2M p = 0.10, 60.3M p = 0.091).

Two bounds are carried, per the pre-registration. The realized arithmetic SD was 0.092 against
the 0.0815 used to size the study, so achieved power for a true +0.133 was **0.93**, not 0.96 —
and the 95% CI still contains +0.133. The claim is therefore that the divergence **does not
survive replication**, not that it is excluded; effects below ≈0.11 are untested, and the
observed +0.065 lies below that.

**What the new seeds did change is the secondary reading.** With 60.3M at n = 8, the arithmetic
mixture's attraction slope becomes **−0.207/decade (p = 0.033**, from −0.139/p = 0.28 at n = 3),
so the scale trend itself reproduces — though only at borderline significance, and §4.2.4 shows
that p is estimator-dependent while the −0.207 estimate is not. But complexity also falls there
(**−0.112/decade, p = 0.012**), and the paired within-model contrast — the correct test, since
both components come from the same models — is **flat: −0.095, CI [−0.314, +0.123], p = 0.36**,
against −0.333 (p = 3.8e-11) in the syntax mixture. The syntax value sits marginally outside the
arithmetic CI. The **dissociation** is thus unreplicated while the **trend** is reproduced; with
n = 3 at the two smaller sizes this is a non-replication, not a refutation. Full readout:
`logs/sv_agreement_probe/replication_60M_results.md`.

### 4.2.4 Does the (3, 3, 8) seed imbalance carry the slope? Estimate no, significance yes

The arithmetic mixture is now unbalanced by construction — 3 seeds at 11.5M, 3 at 23.2M, 8 at
60.3M — because only the 60.3M cell was replicated. OLS over model-level points is unbiased under
imbalance and its SE already accounts for the *x*-distribution, so imbalance alone is not a
problem; what matters is whether imbalance coincides with **unequal residual variance across
sizes**, which is the assumption `linregress` actually makes. It does here: cell SDs are
0.106 / 0.036 / 0.092, a 2.9× spread. So the slope was re-estimated under estimators that drop
that assumption (`scripts/agreement_slope_imbalance_check.py`,
`logs/sv_agreement_probe/slope_imbalance_check.md`).

| estimator (arith, attraction) | slope/decade | p |
|:--|--:|--:|
| OLS — as published | −0.207 | **0.033** |
| HC3 heteroscedasticity-robust SE | −0.207 | **0.073** |
| WLS, weights 1/s²_size | −0.112 | 0.262 |
| group-mean OLS (3 points, df = 1) | −0.237 | 0.407 |
| stratified bootstrap (20k, resampled within size) | −0.208 [−0.350, −0.061] | 0.005 |
| all C(8,3) balanced (3,3,3) subsamples | −0.237 (−0.362 … −0.095) | median 0.066; 43% < 0.05 |

**The point estimate is robust; the p-value is not.** Every estimator that does not reweight the
cells returns −0.21 to −0.24, so the imbalance is not manufacturing the trend. But p ranges from
0.005 to 0.41 depending on assumptions, and the two estimators that are *specifically* right for
this situation disagree with the published one: HC3 gives **p = 0.073**, and a symmetric (3,3,3)
design would have cleared 0.05 only 43% of the time.

**The leverage sits on the under-sampled end, not the over-sampled one.** The intuition that the
8-seed cell distorts the fit is backwards: extra seeds at 60.3M *stabilise* that endpoint
(leverage 0.119/point). The fragile point is **11.5M — leverage 0.254/point, 2.1× a 60.3M point,
and the largest SD (0.106) of the three**. Pooled-variance OLS lends that noisy high-leverage
endpoint the tighter cells' precision, which is why it is the least conservative estimator here;
downweighting it by 1/s² halves the slope outright (−0.112). Under this design the arithmetic
trend rests on three 11.5M seeds.

Two contrasts confirm the diagnosis is about variance structure rather than the method:
the balanced 72-model **syntax** mixture is estimator-invariant (−0.316, p < 1e-10 under OLS,
HC3, WLS and bootstrap alike; SD spread only 1.46×), and the arithmetic **paired** contrast is
flat under all six estimators (p = 0.36 / 0.51 / 0.73 / 0.62 / 0.37, and 0% of balanced
subsamples significant). So the negative half of §4.2.3 — the dissociation not reproducing — is
not fragile at all; only the positive half is.

**Reporting consequence.** The trend is stated as reproducing *in direction and magnitude*, with
p quoted as "0.033 OLS / 0.073 robust" rather than as a bare 0.033. Closing the gap needs seeds at
11.5M, not more at 60.3M.

### 4.3 Trajectory: step 2500 → 5000, paired, n = 6 at 23.2M (same 1,000 sextets)

| quantity | step 2500 | step 5000 | paired change | |
|:--|:--|:--|:--|:--|
| attraction | +0.392 ± 0.096 | +0.303 ± 0.064 | **−0.089 ± 0.086** | **p = 0.053**, 5/6 |
| mismatch cell | 0.465 | 0.535 | **+0.070 ± 0.063** | **p = 0.041**, 5/6 |
| complexity | +0.128 ± 0.058 | +0.143 ± 0.035 | +0.016 ± 0.052 | p = 0.50 |
| match cell | 0.857 | 0.839 | −0.018 | p = 0.44 |
| bare cell | 0.985 | 0.982 | −0.003 | p = 0.74 |

The direction is consistent with the heuristic being unlearned rather than acquired, and the
mismatch cell does improve significantly. But **the attraction component's decline is marginal
at n = 6 (p = 0.053)**, so we report it as suggestive only — and note that n = 6 has only ~53%
power for a decline of this size, so this test could not have settled the question in either
direction (see §6).

Note the contrast with the uncontrolled metric: the balanced BLiMP attractor score rises
+0.060 ± 0.016 over the same window with p = 0.0003 and 6/6 seeds — far stronger than the
isolated attraction effect. Part of that composite gain is the model improving on hard items
generally, not specifically on attraction. This is the same lesson as §5, arriving from a
different direction.

---

## 5. What the standard metric gets wrong

The balanced BLiMP figure at 23.2M (0.383) sits below chance and below the frequency-only
control (0.488) in every model and both paradigms, which reads as a strong positive result:
the model actively prefers the ungrammatical continuation. **Under controlled items that is
true at only one of three scales.**

| size | controlled mismatch accuracy | verdict | balanced BLiMP attractor |
|:--|:--|:--|:--|
| 11.5M | 0.405 ± 0.054 | **below chance** (p < 0.001) | 0.325 (below) |
| 23.2M | 0.511 ± 0.065 | **at chance** (p = 0.435) | 0.378 (below) |
| 60.3M | 0.610 ± 0.055 | **above chance** (p < 0.001) | 0.446 (below) |

The uncontrolled metric reports below-chance behaviour at all three sizes; the controlled design
finds it at one. The gap is item difficulty unrelated to attraction: controlled bare items score
0.982 against BLiMP simple agreement at 0.726 (harder by 0.25 for reasons of relational nouns,
longer spans, and rarer vocabulary), and at 23.2M controlled mismatch scores 0.535 against
BLiMP's 0.383, leaving 0.152 unexplained by attraction and complexity as measured here.

So: anti-syntactic behaviour is real, but it is a property of the **smallest** models, not of
this model class; and the metric that appears to establish it everywhere is measuring
attraction plus difficulty. The defensible headline is the decomposition, not "below chance."

**This part does survive the mixture control**, in its qualitative form only. In the arithmetic
mixture (n = 3/size) mismatch is **0.378 at 11.5M — below chance (p = 0.043)** — and not below
chance at either larger size (0.596 above, p = 0.0009; 0.529 at chance, p = 0.59). So "below
chance only at the smallest size" holds in both mixtures. What does **not** carry over is the
smooth monotone rise: the arithmetic mixture's 60.3M models sit *below* its 23.2M models, the
same non-monotonicity that breaks the scale trend in §4.2.1. Read the below-chance result as a
statement about the smallest models, not as evidence for an orderly scale curve.

*(A pooled figure across all 72 models puts mismatch at 0.509, p = 0.48 — "at chance." That
average is an artifact of combining below-chance 11.5M with above-chance 60.3M models and
should not be quoted.)*

**This is a distinct failure mode from the other two, and worth naming separately.** The
frequency prior (§2) and item difficulty (§5) are **confounds** — a nuisance variable rides along
with the effect, and the fix is to control it, which the crossed design does by construction. The
pooled-72 figure is not a confound: every item is already controlled and every per-size number is
correct. It is an **averaging error across a variable the effect is dissociated on** — pooling
over scale when the quantity *reverses sign* across scale, so the mean lands at the crossing
point and reports "no effect" precisely because the effect is large. No amount of item control
prevents it; the fix is not to pool. The diagnostic is that the pooled estimate is *tight and
null* while the subgroups are far apart — a signature that should trigger disaggregation rather
than a null claim. Notably, the pooled figure would have looked like clean corroboration of the
23.2M "at chance" result (0.509 vs 0.511) while being right for entirely the wrong reason.

---

## 6. Limitations

- **The trajectory result is underpowered, and n = 6 cannot resolve it in principle.** The
  observed paired decline is −0.089 ± 0.086, i.e. dz ≈ 1.04. A paired t-test at n = 6 has
  **~53% power** for an effect that size — a coin flip on whether a real effect of exactly this
  magnitude reaches p < 0.05 at all. Put the other way, the smallest decline this design could
  have detected at 80% power is **≈ 0.125**, larger than the decline that appears to be there.
  So p = 0.053 with 5/6 seeds is the expected reading of a real-but-modest effect at this n, and
  equally the expected reading of nothing; the design does not separate them. Reaching 80% power
  needs **~10 seeds**, 90% needs ~12 (both evaluation-only on existing checkpoint families, or a
  denser checkpoint series). This is why the result is labelled suggestive rather than null: the
  test was not capable of the call either way.
- **No causal evidence.** This is a behavioural signature, not a mechanism. The natural test —
  ablating the components that carry the attractor's number and checking whether agreement is
  restored — was not run.
- **Three sizes.** The scale trend is clean and well-powered per size (n = 24), but a
  three-point log-linear fit should not be extrapolated; we do not project where attraction
  reaches zero.
- **Skill-injected checkpoints; and the headline result rests on a single mixture.** Every model
  scored here comes from a mixture containing injected arithmetic or syntax-enriched text
  (≥80% BabyLM). No 100%-corpus baseline was trained, so "what the corpus alone teaches" is an
  inference, not a measurement. The mixture guard is **unevenly distributed across the three
  results**, and in the opposite direction to their evidential weight:
  - the **n = 6 decomposition (§4.1) and trajectory (§4.3)** span *two* mixtures (20% arithmetic,
    10% syntax-enriched) × 3 seeds, and the families agree to within about one seed SD;
  - the **scale dissociation (§4.2, and the "below chance" scale-dependence in §5)** comes
    entirely from the 72-model set, which is a **single mixture** (10% syntax-enriched).

  So the paper's strongest claim — attraction falling −0.316/decade while complexity stays flat,
  n = 72, p = 2.3e-17 — **splits under replication**: the falling-attraction half reproduces in a
  second mixture (at borderline, estimator-dependent significance), the complexity-stays-flat
  half does not. The scale-dependence of below-chance
  behaviour that §5 rests on does survive (arithmetic 60.3M mismatch 0.626 ± 0.114, above chance,
  p = 0.017). The n = 72 p-value speaks to seed and item variance at one mixture; it
  says nothing about mixture variance. Nothing in the design
  makes a syntax-enriched injection a plausible driver of an attraction × scale interaction
  specifically, and complexity being flat in the same models argues against a generic
  mixture-induced trend — but that was an argument, not a control. **The control has been run
  (§4.2.1–4.2.4), and it resolves the guard unevenly: the trend reproduces, the dissociation does
  not.** In the arithmetic mixture attraction falls −0.207/decade (n = 14; p = 0.033 OLS,
  0.073 heteroscedasticity-robust), but so does complexity (−0.112/decade, p = 0.012 / 0.055),
  leaving the paired attraction−complexity contrast flat (−0.095, p = 0.36 — flat under every
  estimator) where the syntax mixture gives −0.333 (p = 3.8e-11). **So "attraction shrinks with
  scale" is now a two-mixture result at borderline significance in the second mixture; "and
  complexity does not" remains a single-mixture one, and that is the claim that stays labelled as
  such.**
  Four addenda, two of them now closed: (i) the power estimate that justified the first attempt
  at n = 3 was optimistic — it used the syntax mixture's within-size SD (0.063) where the
  arithmetic mixture's is 0.094, giving realized power of 0.62 rather than 0.90, and only 0.21
  for the gap actually observed; **the lesson is to take the SD from the arm being measured, not
  the arm being replicated from**; (ii) the mixtures also differ in natural-language volume, which
  would have confounded a genuine English-exposure effect with mixture identity — **tested and
  excluded on four grounds (§4.2.2)**, most directly because the arithmetic mixture's English
  perplexity handicap vanishes by 60.3M (−0.3%, p = 0.61) precisely where attraction diverged
  most; (iii) the 60.3M divergence itself — the one anomalous cell — **did not survive
  replication at n = 8** (+0.065, p = 0.091, halved from +0.133; §4.2.3), with the pooled
  interaction weakening to p = 0.140. That null carries its own bound: achieved power was 0.93
  for the original effect (realized SD 0.092 vs the 0.0815 assumed), the CI still contains
  +0.133, and effects below ≈0.11 are untested — so the divergence is *unreplicated*, not
  *excluded*; **(iv) the replication left the arithmetic mixture unbalanced (3, 3, 8), and the
  second-mixture slope's significance turns out to depend on the estimator — p = 0.033 under
  pooled-variance OLS but 0.073 under a heteroscedasticity-robust SE, with the −0.21 point
  estimate stable throughout (§4.2.4). The high-leverage cell is the under-sampled 11.5M end
  (leverage 0.254/point, SD 0.106), so the arithmetic trend effectively rests on three seeds;
  additional seeds would be worth spending at 11.5M, not at 60.3M.**
- **Templates are simpler than natural text.** Controlled items use high-frequency vocabulary
  and short frames; absolute levels (bare = 0.98) are not comparable to natural-corpus
  performance — only the *contrasts* are.
- **One corpus, one budget.** All results are at ~10M words / 5,000 steps. The parameter axis is
  varied; the data axis is not, so "scale" here means parameters at fixed data.

---

## 7. Relation to prior work

Agreement attraction in neural LMs is not new: reported for LSTMs by Linzen et al. (2016),
extended by Marvin & Linzen (2018) and Gulordava et al. (2018), localized to a small set of
"number units" by Lakretz et al. (2019), and a standing weak spot in BLiMP evaluations of small
models. What this note adds is narrower and largely methodological:

1. **A frequency-exact decomposition.** Prior reports rest on item sets where the attractor
   always mismatches, which cannot separate attraction from item difficulty. The crossed design
   separates them and puts numbers on both.
2. **A dissociation under scale — in one training mixture.** Attraction falls −0.316/decade
   while complexity stays flat (n = 24 per size), which would put scale on the syntactic
   component specifically rather than on general tolerance for intervening material. A
   second-mixture control (§4.2.1–4.2.4) splits this: the **falling-attraction half reproduces**
   (−0.207/decade, n = 14; p = 0.033 OLS / 0.073 robust), the **complexity-flat half does not** —
   complexity falls there too, leaving the paired contrast flat under every estimator. So
   "attraction shrinks with scale" is offered as a two-mixture observation at borderline
   significance in the second, and the *dissociation* as a single-mixture one; neither is an
   established scaling law.
3. **A cautionary result about the standard metric.** The same weights read as "poor" under raw
   accuracy, "below chance and anti-syntactic" under the number-balanced metric, and "at chance
   with a large attraction effect" under the controlled design — and the balanced metric reports
   below-chance behaviour at three sizes where the controlled design finds it at one. Three
   metrics, three qualitatively different conclusions; the most striking one is the least
   correct. A fourth reading — pooling the controlled metric over all 72 models — recovers a
   confident "at chance" from a sign reversal, a failure of aggregation rather than of item
   control (§5).

---

## 8. Methodological byproduct: what makes a skill usable in a delayed-exposure design

This analysis began as a search for a second skill for a critical-period study (does delaying a
skill's first exposure cap its final attainment?). Simple subject–verb agreement was the
candidate and was **rejected on two independent grounds**, both general.

**(i) The skill must be near-absent from the base corpus, or it cannot be delayed.** Delay is
implemented by withholding skill-bearing lines from the early training stream. That costs
**1.51%** of lines for wh-object-gap extraction (16,030 / 1,058,740). For subject–verb agreement
it costs **39.0%** (412,474 lines) — a strict lower bound, counting only overt number-marked
auxiliaries; including third-person-singular `-s` lexical verbs pushes it past half. Withholding
that much is not a timing manipulation but a corpus substitution, leaving a register-skewed
residue (past-tense narrative, fragments, imperatives). Synthetic arithmetic is delayable because
it is absent from the corpus; wh-extraction is delayable at 1.5%. **Ambient skills are not
delayable by subtraction.**

**(ii) The post-onset window must be insufficient for the skill, or there is nothing to detect.**
The predictor of a delay deficit is time-to-plateau relative to budget. Simple agreement reaches
**95.3%** of its final headroom by step 2500 of 5000; the whole second half adds +0.011 ± 0.016
(p = 0.17). The arithmetic skill used in the main study reaches only 74.2% by step 2500, gains
+0.206 over the second half, and shows a delay deficit of 0.214. A skill that finishes learning
before the delay window opens has nothing for a delay to disturb; the predicted deficit for
agreement is ~0.01 against a detection floor of 0.05 at n = 3.

Neither ground is specific to agreement. Together they give a cheap pre-screen for any candidate
skill — **corpus prevalence below a couple of percent, and time-to-plateau well short of the
training budget** — both measurable without training a single model.

---

## Artifacts

| what | where |
|:--|:--|
| BLiMP balanced probe + frequency-only control | `scripts/sv_agreement_probe.py`, `logs/sv_agreement_probe/results.json` |
| saturation + composite scale/trajectory | `scripts/sv_agreement_saturation.py`, `saturation_scale.json` |
| crossed quartet — 23.2M detail | `scripts/agreement_attraction_quartet.py`, `quartet.json` |
| crossed quartet — 72-model scale set | `quartet_scale.json` |
| crossed quartet — 9-model arith-mixture control | `quartet_scale_arithmix.json` |
| mixture replication analysis | `scripts/agreement_mixture_replication.py`, `mixture_replication.{json,md}` |
| English-exposure re-analysis | `scripts/agreement_exposure_recheck.py`, `exposure_recheck.{json,md}` |
| 60.3M replication, n = 8 (pre-registered read) | `scripts/agreement_60M_replication_readout.py`, `replication_60M_{readout.json,results.md}` |
| seed-imbalance robustness of the slope tests | `scripts/agreement_slope_imbalance_check.py`, `slope_imbalance_check.{json,md}` |
| crossed quartet — steps 2500 / 5000 | `quartet_step2500.json`, `quartet_step5000.json` |
| full running log incl. corrections | `logs/sv_agreement_probe/results.md` |
