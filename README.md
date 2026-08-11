# Can LLM Critical Periods Be Reopened?

Small language models trained from scratch under controlled skill-exposure schedules,
testing whether a delayed start on a skill leaves a permanent deficit, and whether any
intervention can undo it.

Every model here is trained from random initialization. No pretrained checkpoints are
used anywhere, including for baselines, because a public checkpoint has already passed
whatever critical period it has on uncontrolled data.

## The question

A *critical period* for skill S ends at step T if a model first exposed to S after T
reaches a lower final accuracy than a model exposed from the start, **at matched total
S-token exposure**, and the gap survives further training. Only the timing of exposure
differs between conditions, never the amount.

The study runs in three phases:

1. **Detect.** Does a timing-induced cap exist at all?
2. **Reopen.** Can an intervention undo it without trading away general language ability?
3. **Diagnose.** Does a recovered model repair the original circuit, or build a new one?

Recovery that costs general-language perplexity is scored as forgetting, not reopening,
so skill accuracy and held-out perplexity are logged together at every checkpoint.

## Findings

Every claim below carries exactly one evidential label. Definitions are at the bottom.
Numbers are absolute endpoints with cross-seed SD at n = 3. Recovery is never reported
as a percentage of the gap: that denominator is the seed-variable null, so the ratio
moves even when the underlying result does not.

### Detect: arithmetic has a critical period

Two-to-four digit addition, exact-match accuracy at step 5000.

| size | early exposure | delayed (d50) | deficit | Holm p |
|:--|:--|:--|:--|--:|
| 23M | 0.791 +/- 0.007 | 0.577 +/- 0.051 | 0.214 +/- 0.055 | 0.0107 |
| 60M | 0.829 +/- 0.027 | 0.600 +/- 0.040 | 0.229 +/- 0.037 | 0.0087 |

Delayed exposure produces a persistent endpoint deficit at both confirmatory sizes
(**ESTABLISHED**), of similar magnitude at each (**ESTABLISHED**, with the caveat that
only size effects larger than 0.144 are excluded at 80% power). Severity is seed-variable
and driven by the delayed endpoint rather than the early anchor (**ESTABLISHED**).

11M is reported but excluded from the confirmatory set by a pre-registered capacity-floor
rule, so its nominally significant deficit is not a claim.

### Detect: long-distance syntax does not

Long-distance wh-licensing shows no detectable critical period under five distinct
probes: mean endpoint, learning dynamics, acquisition rate, representational probing,
and LR budget (**ESTABLISHED null**, scoped to <= 60.3M parameters and the 10M-word
BabyLM corpus). These are five distinct routes to the same negative, not one result
confirmed five times.

A probe positive control passes, so the structure is encoded; it is the readout that is
timing-invariant (**ESTABLISHED**).

This inverts the pre-registered prediction, which expected syntax to carry the stronger
critical period. Arithmetic carries Detect alone.

The dissociation is **ESTABLISHED (scoped)**: delayed exposure costs arithmetic 0.214
and costs long-distance syntax nothing measurable. The scope matters. Syntax has 0.028
of headroom against arithmetic's 0.791, and the syntax MDE (0.048) exceeds its total
headroom, so the *absolute* dissociation is established while the differential-
*susceptibility* reading is not.

### Reopen: one intervention of four works

All arms fork from a single bit-identical checkpoint at step 2500 and reconstruct the
same data stream, so endpoints are comparable across arms.

| size | treated (lr_shift) | delayed null | early anchor | gate (0.90x anchor) | result |
|:--|:--|:--|:--|--:|:--|
| 23M | 0.767 +/- 0.014 | 0.577 +/- 0.051 | 0.791 +/- 0.007 | 0.712 | PASS |
| 60M | 0.796 +/- 0.022 | 0.600 +/- 0.040 | 0.829 +/- 0.027 | 0.746 | PASS |

An LR-shift intervention reopens the deficit at both sizes (**ESTABLISHED**), to a
scale-invariant endpoint (**ESTABLISHED**). It also collapses cross-seed variance:
the null spans 0.051-0.061 SD, the treated arms 0.014-0.022 (**ESTABLISHED**).

Closure at d50 is therefore largely a learning-rate-schedule effect rather than
representational entrenchment (**ESTABLISHED**, from the cram contrast: cram+lr_shift
is approximately lr_shift, and both greatly exceed cram alone).

The other three Tier-1 interventions, at n = 1 against the null's cross-seed range
[0.524, 0.626]:

| arm | endpoint | reading |
|:--|--:|:--|
| reset (embedding) | 0.586 | no detectable recovery |
| reset + spike | 0.576 | no detectable recovery |
| LR spike | 0.536 | no detectable recovery |
| cram | 0.574 | no detectable recovery |
| freeze | 0.332 | harmful, below the entire null range |

All **SUGGESTIVE (n = 1)**. One seed cannot support "fails to reopen".

The 60M forgetting gate trips (+1.670 +/- 0.104 ppl against a +1.0 gate), but a matched-LR
control shows a larger lead (+2.477 +/- 0.240), so the residual is
**BOUNDED-NOT-EXCLUDED**, bounded above by 1.67 ppl with no point estimate.

### Diagnose: circuit identity is a continuum

By mean-ablation transfer, recovered models span a continuum from near-complete transfer
of the original circuit (0.868) to full reorganization (0.187) (**ESTABLISHED,
single-method**). 60M reorganizes at every seed (**ESTABLISHED, single-method**).

Whether deficit depth grades circuit identity is **BOUNDED-NOT-EXCLUDED**
(Spearman rho = -0.71, exact permutation p = 0.068, and leave-one-out fragile: four of
six single-point deletions push it above 0.05).

**This is a single-method result and is reported as one.** Two candidate second methods
were tested across all six triads and neither discriminates:

- **CKA** stays at 0.66-0.82 for every triad regardless of what ablation says, because
  shared initialization keeps global geometry similar even where causal wiring differs.
- **Single-head activation patching** is flat (0.59-0.79, SD 0.083) against ablation
  transfer's 0.19-0.87 (SD 0.256), with zero correlation to it (r = 0.045, p = 0.93).

Both were retracted as corroboration. A second method only corroborates if it can be
shown to return a *different* answer on the cases the first method separates.

## Design

- **Skills.** Arithmetic (2-4 digit addition, synthetic, infinite supply, exactly
  gradeable) and long-distance wh-question object-gap extraction. Word-frequency
  prediction serves as a negative control that should show no critical period.
- **Syntax metric.** The wh-licensing *interaction*, not raw paradigm accuracy. Raw
  BLiMP pairs differ by a single word, so the score is dominated by the unigram prior:
  an untrained baseline scored 0.000 and 1.000 on the two halves with no syntactic
  knowledge at all. The interaction cancels the wh-word main effect, placing a purely
  frequency-driven model at exactly 0.500.
- **Corpus.** BabyLM strict-small (about 10M words), with synthetic arithmetic layered
  on a controlled delay schedule.
- **Matched exposure.** Delay conditions differ only in *when* skill data enters the
  stream, never in how much.
- **Checkpoints.** Fixed step intervals, never wall-clock. Delay onsets and intervention
  triggers resolve to exact step numbers before training starts, so runs reproduce
  regardless of thermal throttling.

## Architecture

| size | layers | width | heads | params |
|:--|--:|--:|--:|--:|
| 11M | 4 | 256 | 8 | 11.5M |
| 23M | 6 | 384 | 8 | 23.2M |
| 60M | 8 | 640 | 10 | 60.3M |

Context 512, vocab 16384 (custom BPE trained on BabyLM itself, not GPT-2's 50K, which
would waste embedding capacity at this corpus size). RoPE positions, pre-norm, GELU.
AdamW with cosine decay and warmup, 5000 steps.

**Embedding and output weights are untied on purpose.** Resetting the embedding layer
should affect input representations only, not the output head, so recovery can be
attributed to one or the other instead of being conflated.

## Repository layout

```
reopen/            core package: model, training loop, data pipeline,
                   delay scheduler, interventions, eval harness, diagnosis
reopen/skills/     arithmetic, syntax, and vocab-frequency generators
scripts/           experiment drivers, batch runners, analysis, paper build
notes/             pre-registrations, working notes, the Week 9 stats backbone
logs/              per-checkpoint eval results and full batch run logs
data/              tokenizer, BLiMP paradigms, eval sets, tokenized memmaps
paper/             manuscript source and compiled PDFs
```

`notes/week9-stats-backbone.md` is authoritative for the write-up: it holds the claim
ledger, one evidential label per claim, the multiplicity families, and the map from each
number to the artifact that owns it.

## Reproducing

Requires an NVIDIA GPU (developed on an RTX 4060, 8GB).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Fetch and prepare the corpus (it is not redistributed here, see below):

```bash
python scripts/prepare_babylm.py --vocab-size 16384
python scripts/prepare_skills.py
```

Train a baseline and a delayed run:

```bash
python scripts/train_baseline.py --model-size 23M --skill arith --delay-onset 0.0 --seed 1
python scripts/train_baseline.py --model-size 23M --skill arith --delay-onset 0.5 --seed 1
```

Fork the delayed run at its common checkpoint and apply the intervention:

```bash
python scripts/run_intervention.py \
    --from-run arith_delay50_exp20_size23M_seed1 \
    --fork-step 2500 --intervention lr_shift --shift-source-start 0
```

Recompute the statistics:

```bash
python scripts/week9_stats_backbone.py
```

Checkpoints land in `checkpoints/{run_id}/step_{step:07d}.pt`, each storing weights,
optimizer state, step number, and a metadata dict recording skill, delay onset, model
size, seed, and any intervention applied.

## What is not in this repository

| excluded | size | why |
|:--|--:|:--|
| `checkpoints/` | ~98 GB | Far past any remote's limits. Training is deterministic and resume-validated, so these regenerate. |
| `data/raw/` | ~910 MB | Two files exceed GitHub's 100 MB hard limit, and the corpus is third-party with mixed upstream licences. Fetch it with `prepare_babylm.py`. |
| `.venv/` | ~4.5 GB | Platform-specific binaries. |

Everything else ships, including the tokenized `.bin` memmaps the runs actually consumed.
Eval numbers are comparable only between models trained on the same `vocab.bin`, so that
file is pinned here rather than regenerated.

## Evidential labels

| label | meaning |
|:--|:--|
| **ESTABLISHED** | Replicated at n >= 3, direction stable in every replicate, decided by a rule fixed before the data, no unresolved confound. |
| **SUGGESTIVE** | Direction consistent, but at least one of: n < 3, single method, single mixture, observational factor, or estimator-dependent significance. |
| **BOUNDED-NOT-EXCLUDED** | A null or attenuated result whose interval still contains the effect it failed to confirm, or a cost bounded above but not point-estimated. Always quoted with its bound. |

Labels are fixed once in the claim ledger and must match across the abstract, the
relevant section, and the discussion.

## Status

Analysis complete, manuscript in preparation. Two versions build from the same source
with no results dropped from either.
