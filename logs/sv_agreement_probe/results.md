# Subject-verb agreement — headroom probe on the 10M-word corpus at 23M

**Question.** Before investing in a delay scheduler + seed infrastructure for a new skill:
does subject-verb agreement clear a real ceiling on *this* corpus at 23M — the way local
wh-licensing did (0.604) — or sit at floor the way long-distance did (0.534)?

**Cost.** Zero training. Agreement was never an injected skill, so existing delay-0
checkpoints measure exactly what the corpus alone teaches. Six models: `arith_delay0_exp20`
and `syntax_delay0_exp10`, 23M, seeds 1–3 each (two families = robustness against the
injection mixture driving the result). Script `scripts/sv_agreement_probe.py`.

## The frequency confound is REAL here — raw accuracy is not usable

Every BLiMP agreement pair differs in exactly **one word** (the verb form: `hasn't`/`haven't`,
`references`/`reference`). That is structurally the same trap as `wh_vs_that`, where the score
was dominated by the unigram prior for "that". Confirmed by an explicit **unigram-only control**
(sentences scored by summed unigram log-prob from `train.bin` — no context, no syntax; this is
an exact frequency baseline, stronger than the step-100 proxy used for wh, and no step-100
checkpoint survives on disk):

| paradigm | control RAW | control on singular-subj | control on plural-subj | control BALANCED |
|:--|:--:|:--:|:--:|:--:|
| distractor_agreement_relational_noun | 0.471 | 0.105 | 0.830 | **0.467** |
| distractor_agreement_relative_clause | 0.527 | 0.122 | 0.897 | **0.510** |
| regular_plural_sv_agreement_1 | 0.411 | 0.182 | 0.908 | **0.545** |
| regular_plural_sv_agreement_2 | 0.472 | 0.094 | 0.911 | **0.503** |
| irregular_plural_sv_agreement_1 | 0.519 | 0.235 | 0.813 | **0.524** |

A pure frequency model scores **0.09–0.24 on singular-subject items and 0.81–0.91 on
plural-subject items** — a massive split, because bare/plural verb forms (`have`, `reference`,
`don't`) are far more frequent than their `-s` singular counterparts. Any raw agreement number
is therefore mostly measuring that prior.

**Fix — number-balanced interaction** = mean(acc on singular-subject items, acc on
plural-subject items), the direct analogue of `wh_licensing_interaction()`. A model that always
prefers one number form scores 1.0 on one subset and 0.0 on the other ⇒ exactly 0.5. **Validated:
the unigram control collapses to 0.488 (attractor) / 0.524 (no-attractor)** — i.e. ≈ chance, the
same way the step-100 model landed at exactly 0.500 on the wh interaction. All numbers below are
this balanced metric.

## RESULT — the answer splits by construction

| condition | balanced interaction | vs references |
|:--|:--|:--|
| **no-attractor (simple) agreement** | **0.726 ± 0.020** [0.699, 0.754] | **above** wh local (0.604) |
| **attractor (Linzen-style) agreement** | **0.383 ± 0.042** [0.347, 0.465] | **below chance (0.500)** |
| — unigram control | 0.524 / 0.488 | ≈ chance ✓ |
| — wh local / long-distance | 0.604 / 0.534 | prior syntax results |

n = 6 models (2 training families × 3 seeds).

### Simple agreement: PASSES, with the best headroom this project has seen in syntax
0.726 is the highest syntax-family number produced so far — above local wh-licensing (0.604) and
well clear of the LD floor (0.534). Critically, **seed variance is tiny: SD 0.020**, and it holds
across both training families (arith-injected 0.739, syntax-injected 0.711). This matters more
than the level: the fatal problem with wh LD was that baseline seed variance (d0 ranging
0.520–0.586) *exceeded the entire effect being measured*. Here it does not.

### Attractor agreement: FAILS, and in a more interesting way than LD wh did
0.383 is **below chance and below the unigram control (0.488)** — every model, both attractor
paradigms, all six seeds. This is not "no signal" (LD wh's failure mode, sitting at floor); it is
a systematic *wrong* heuristic: the model agrees with the linearly nearest noun (the attractor)
rather than the syntactic head. That is the classic Linzen agreement-attraction error reproduced
at 23M. Because the frequency control sits at chance here, the below-chance score is genuinely
anti-syntactic behaviour, not an artifact.

## Read for the go/no-go decision

- **The construction originally proposed (attractor-based, Linzen-style) does NOT clear a
  ceiling** ⇒ it is a third data point in the "small corpus, low headroom" pattern, with a
  distinct failure mode (learned recency heuristic, not absence of signal). Do not build a
  Detect grid on it.
- **Simple (non-attractor) agreement does clear a ceiling** and has the seed stability wh LD
  lacked ⇒ it is a viable Detect candidate.
- **But its usable dynamic range is ~0.23** (0.726 vs a 0.5 floor), against arithmetic's ~0.79
  (0.79 vs a ~0.0 floor) — roughly 4× smaller. With SD 0.020, a delay effect of ≥ 0.05 would be
  ~2.5 SD and detectable at n=3; anything smaller would not be. That is the power constraint any
  pre-registration has to accept up front.
- **Trade-off to weigh:** simple agreement is an *easier* skill than the one we wanted to test,
  and easy skills may be robust to exposure timing — so passing the headroom gate does not imply
  a critical period will be found.

## Caveats
- These are skill-injected models (80% / 90% BabyLM), not 100%-corpus models. Both families agree
  (0.739 vs 0.711, difference ≈ the SD), so the mixture is unlikely to be driving the result, but
  a true standard-exposure baseline was not trained for this probe.
- `irregular_plural_subject_verb_agreement_2` was dropped: it varies the **subject noun**
  (`women`/`woman`) and the `-s` classifier does not handle irregular plurals. `irregular_..._1`
  already covers irregular agreement. Note the `_2` variants generally vary the noun rather than
  the verb, so their balancing cancels noun-number frequency instead of verb-form frequency —
  both are valid, and the control confirms cancellation works for both (0.503, 0.524).
- Whole-sentence log-prob scoring, matching the Week-2 wh numbers (0.604 / 0.534) for
  comparability. The continuation-scored wh figures (local ~0.68, LD ~0.586) come from a
  different metric and are **not** directly comparable to these.

Artifacts: `results.json`, script `scripts/sv_agreement_probe.py`, data
`data/blimp/{distractor_agreement_*,{ir,}regular_plural_subject_verb_agreement_*}.jsonl`.

---

# ADDENDUM (2026-07-26) — saturation, scale, and the delayability blocker

Three follow-ups, all **zero training**, on checkpoints already on disk.
Script: `scripts/sv_agreement_saturation.py`; data `saturation_scale.json`.

## 1. DELAYABILITY — simple agreement cannot be delayed by this pipeline

The delay mechanism is line-level corpus withholding (`syntax.filter_corpus_files`:
target-bearing lines are pulled from the early stream, injected later). Cost of that
manipulation, measured on the strict-small corpus:

| skill | lines withheld | % of corpus |
|:--|--:|--:|
| wh object-gap extraction | 16,030 / 1,058,740 | **1.51%** |
| subject-verb agreement (overt number-marked auxiliary only) | 412,474 / 1,058,740 | **39.0%** |

39.0% is a **strict lower bound** — auxiliaries only (`is/are/was/were/has/have/does/do`
+ negations); adding 3sg `-s` lexical verbs pushes it past half. Withholding 39-60% of the
corpus for the first half of training is not a timing manipulation, it is a different
corpus, with a register-skewed residue (past-tense narrative, fragments, imperatives).
This violates the "total exposure matched, only timing differs" premise.

**Arithmetic is delayable because it is absent from BabyLM; wh is delayable at 1.5%.
Agreement is ambient, and ambient skills are not delayable by subtraction.**

## 2. SATURATION — the post-onset window is already sufficient

The quantity that predicts a delay deficit is **sufficiency of the post-onset window**:
under matched exposure a d50 model must acquire the skill entirely within the second half.

| | @2500 | @5000 | second half adds | % of headroom by 2500 |
|:--|:--|:--|:--|:--|
| simple agreement (n=6) | 0.7151 ± 0.0097 | 0.7256 ± 0.0204 | **+0.0105 ± 0.0160** (p=0.17, 4/6 seeds) | **95.3%** |
| arithmetic exact-match | 0.592 | 0.798 | **+0.2060** | 74.2% |

Arithmetic was still climbing at step 3500-4000, so a 2500-step window was *insufficient*
=> observed d50 deficit 0.214. Simple agreement is finished by step 2500, so a 2500-step
window at double density is **more than sufficient** => **predicted d50 deficit ~0.01
(range 0.00-0.04), against a 0.05 detection floor. Underpowered by ~5x.**

Caveat: this predicts the insufficient-window / LR-schedule component, not a pure
entrenchment component. But Week 4 established arithmetic's d50 deficit is *largely*
LR-schedule closure (lr_shift recovers to 0.767), i.e. entrenchment is the empirically
small component in this setup.

Literature is consistent but indirect — no one has run delayed exposure on agreement.
Data-scale / acquisition-order work (Zhang et al. 2021; Perez-Mayos et al. 2021; Huebner
et al. 2021 BabyBERTa; Warstadt et al. 2020; Evanson, Lakretz & King 2023) converges on
agreement being among the earliest and cheapest BLiMP phenomena. Weaker evidence than the
in-domain +0.0105; do not lead with it.

**=> DO NOT build a Detect grid on simple agreement. Two independent kills:
not implementable, and predicted effect 5x under floor.**

## 3. SCALE — a clean dissociation, n=24 seeds x 3 sizes (size sweep, free)

| size | attractor (balanced) | simple (balanced) | below chance |
|:--|:--|:--|:--|
| 11.5M | 0.325 ± 0.030  (t=-28.6, p=1.8e-19) | 0.701 ± 0.011 | 24/24 |
| 23.2M | 0.378 ± 0.024  (t=-24.6, p=5.1e-18) | 0.710 ± 0.016 | 24/24 |
| 60.3M | 0.446 ± 0.029  (t= -9.1, p=4.6e-09) | 0.712 ± 0.014 | 23/24 |

- **attractor vs log10(params): slope +0.167/decade, r=0.874, p=1.1e-23**;
  11M->60M (5.2x) **+0.121, Welch p=2.6e-18**
- **simple vs log10(params): slope +0.015/decade** (r=0.303, p=0.0096) — statistically
  detectable but **11x smaller**, and the total 11M->60M change (+0.011) is one seed SD.
  Effectively saturated on the scale axis too.
- **71 / 72 models below chance on attractor agreement.**
- Linear-in-log extrapolation crosses chance at ~126M params. **Illustrative only**
  (3 sizes, linear fit, no guarantee the trend is log-linear out of range) — do not quote
  as a prediction.

## 4. TRAJECTORY — the heuristic is being UNLEARNED, not acquired

Attractor balanced interaction rises **+0.0603 ± 0.0160 over the second half of training,
6/6 seeds, paired p=0.0003** (0.323 -> 0.383).

This **corrects the framing in the main write-up above.** "The model has learned a
systematic wrong heuristic" is right about the *state* and wrong about the *trajectory*:
the recency heuristic dominates early acquisition and is slowly being overwritten, but has
not crossed chance by the end of a 10M-word budget. It also weakens with scale (§3).
Correct claim: **a recency heuristic that dominates early acquisition and is not yet
overwritten at 10M words / 60M params.**

## 5. What the attraction finding still needs before it is citable

**The load-bearing missing control.** Every BLiMP distractor item has the attractor
*mismatching* the head — there is no matched condition where the attractor agrees. So
"below chance on attractor items" remains confounded with "these items are longer / contain
a PP or RC." If the model is below chance on all complex agreement items regardless of
attractor number, the linear-proximity claim dies and it is a length/complexity effect.
The existing sg/pl asymmetry does **not** settle this: the unigram control pushes the same
direction (control sg 0.105 / pl 0.830), so the asymmetry is frequency-confounded.

**Fix:** template-generated quartet crossing head number x attractor number in a fixed
lexical frame (zero training, ~2k items). This is the same shape of trap as `wh_vs_that`
and the raw-accuracy frequency confound — run it before writing.

Secondary gaps: all 6 primary models are skill-injected (80/90% BabyLM), no 100%-corpus
baseline; the two families agree (0.739 vs 0.711) which bounds but does not remove this.

**Novelty positioning (honest):** attraction in LMs is known (Linzen et al. 2016; Marvin &
Linzen 2018; Lakretz et al. 2019). Novel here: (a) *below chance* under a frequency-
controlled metric, where raw accuracy 0.30-0.53 would read merely as "poor"; (b) the
within-model dissociation against 0.726 simple agreement; (c) the unlearning trajectory;
(d) the log-linear scale recovery at n=24/size. That is a workshop note or a strong paper
section — **not a standalone conference paper.**

---

# QUARTET CONTROL — RESULT (2026-07-26). Verdict (c) MIXED, with a specific structure.

The blocking control from §5 above, run. Script `scripts/agreement_attraction_quartet.py`,
data `quartet.json`, console `quartet.out`. 1000 fully-crossed sextets (12,000 sentences),
6 models (2 families x 3 seeds, 23M). **Zero training.**

## Design validation (both checks passed before reading any model number)
- All 6 verb pairs tokenize to **equal token counts** (is/are, was/were, has/have, does/do,
  isn't/aren't, wasn't/weren't) — unequal counts would length-bias summed log-prob, so pairs
  failing this are dropped by construction. None failed. 0 tokenizer-unclean cells.
- **Unigram-only control returns attraction exactly +0.0000** (and complexity +0.0000).
  This is the analytic guarantee of the design: within a head number, match vs mismatch has
  the IDENTICAL verb pair and IDENTICAL correct answer, so the verb-form frequency prior —
  the thing that wrecked raw BLiMP accuracy — cancels *exactly*, not on average.

## Results (PRIMARY = verb-position log-prob, n=6)

| cell | balanced accuracy |
|:--|:--|
| bare ("The farmer is …") | **0.982 ± 0.014** |
| attractor MATCHES head number | **0.839 ± 0.033**  (vs chance p=2.0e-06) |
| attractor MISMATCHES head number | **0.535 ± 0.071**  (vs chance **p=0.275 — AT chance**) |

| component | value | |
|:--|:--|:--|
| **ATTRACTION** = match − mismatch | **+0.303 ± 0.064** | t=11.52, **p=8.6e-05**; positive 6/6 in BOTH head-number directions (sg +0.255, pl +0.352) |
| **COMPLEXITY** = bare − match | **+0.143 ± 0.035** | p=1.8e-04 |

- **Attraction is 2.12x the complexity cost**, and larger in 6/6 models (paired p=0.0047).
- Present in both frame types (PP 6/6, object-RC 6/6) with **no consistent ordering**
  (paired diff +0.019 ± 0.240, p=0.85) — so it is not an artifact of one construction.

## What this settles, and what it kills

**CONFIRMED — the linear-proximity heuristic is real.** This is what the control existed to
test, and it passes cleanly: a frequency-exact contrast, sign-consistent across both number
directions, in both frame types, at 2x the complexity cost. The confound raised in §5 — that
"below chance on attractor items" might be pure length/complexity — is **refuted**: complexity
is real but is the *smaller* component. The finding is not an item-difficulty artifact.

**KILLED — the "below chance / systematically anti-syntactic" framing.** Under matched
conditions, mismatch items land **at chance (0.535, p=0.275), not below it.** The BLiMP
figure of 0.383 therefore reflects attraction **plus** difficulty specific to BLiMP's
distractor items (relational nouns, longer spans, rarer vocabulary), not a clean measure of
anti-syntactic behaviour. Reconciliation: quartet bare 0.982 vs BLiMP simple 0.726 (BLiMP
items harder by 0.25); quartet mismatch 0.535 vs BLiMP attractor 0.383 (0.152 unexplained by
attraction + complexity as measured here).

**=> Do NOT lead with "below chance."** The defensible headline is the DECOMPOSITION:
*attraction +0.303 vs complexity +0.143, frequency-exact, n=6* — plus the scale curve
(+0.167/decade) and the unlearning trajectory (+0.060 over the second half). The strong claim
("the model is anti-syntactic") does not survive; the mechanistically specific claim
("a linear-proximity heuristic dominates, and it is not merely item difficulty") does.

Per the project's cross-method rule: verb-position and whole-sentence scoring are two
scorings of ONE probe (whole-sentence attraction +0.337 mean, same direction) and are **not**
reported as mutual corroboration.

---

# QUARTET EXTENDED TO SCALE + TRAJECTORY (2026-07-26) — two conclusions changed

Ran the crossed design across the 72-model scale set and at steps 2500/5000, to replace the
composite-BLiMP versions of findings 2 and 3 with the isolated decomposition.
`quartet_scale.json`, `quartet_step2500.json`, `quartet_step5000.json`.

## Scale (n=24 seeds/size, 400 sextets) — finding UPGRADED

| size | bare | match | mismatch | ATTRACTION | complexity |
|:--|:--|:--|:--|:--|:--|
| 11.5M | 0.969±0.019 | 0.835±0.045 | **0.405±0.054** | **+0.430±0.071** | +0.134±0.043 |
| 23.2M | 0.956±0.036 | 0.825±0.066 | **0.511±0.065** | **+0.314±0.081** | +0.131±0.057 |
| 60.3M | 0.956±0.030 | 0.811±0.065 | **0.610±0.055** | **+0.201±0.056** | +0.145±0.055 |

- attraction **−0.316/decade** (r=−0.803, p=2.3e-17); 11.5M→60.3M **−0.229**, Welch p=5.6e-16
- complexity **+0.016/decade (p=0.43) = FLAT**; 11.5M→60.3M +0.011, p=0.44
- attraction>0 in **72/72**; attraction>complexity 24/24, 22/24, 17/24 (3.20x/2.39x/1.38x)

**Scale acts specifically on the syntactic component.** Because complexity is flat, this also
rules out "bigger models are just better at long sentences." This is now the strongest result.

## Trajectory (paired, n=6 at 23.2M, same 1000 sextets) — finding DOWNGRADED

| quantity | 2500 | 5000 | paired change | |
|:--|:--|:--|:--|:--|
| attraction | +0.392±0.096 | +0.303±0.064 | −0.089±0.086 | **p=0.053**, 5/6 |
| mismatch cell | 0.465 | 0.535 | +0.070±0.063 | p=0.041, 5/6 |
| complexity | +0.128±0.058 | +0.143±0.035 | +0.016±0.052 | p=0.50 |
| match / bare | 0.857 / 0.985 | 0.839 / 0.982 | −0.018 / −0.003 | p=0.44 / 0.74 |

The composite BLiMP metric gave +0.060, **p=0.0003, 6/6** over the same window — much stronger
than the isolated effect. Part of the composite gain is general improvement on hard items, not
attraction. **Unlearning is now SUGGESTIVE ONLY (p=0.053), not established.**

## The "below chance" retraction is REFINED, not total

Controlled mismatch accuracy vs chance: **11.5M 0.405 BELOW (p<0.001) | 23.2M 0.511 AT
(p=0.435) | 60.3M 0.610 ABOVE (p<0.001)**. The uncontrolled balanced BLiMP metric reports
below-chance at **all three** sizes (0.325/0.378/0.446).

So anti-syntactic behaviour is **real at the smallest scale only** — the earlier flat statement
"below chance is dead" was itself too strong; it is dead *at 23.2M*, where it was first measured,
and true at 11.5M. Do not quote the pooled 72-model mismatch (0.509, p=0.48) as "at chance" —
that average combines below-chance 11.5M with above-chance 60.3M.

**Write-up finalized at revised scope: `notes/agreement-attraction-writeup.md`.**
