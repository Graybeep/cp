# PROJECT: Can LLM Critical Periods Be Reopened?

## Quick Context
Testing whether small LLMs have "critical periods" for specific skills (arithmetic, syntax) during pretraining — windows after which the skill becomes hard to learn no matter how much more data is added — and whether targeted interventions can reopen that window. Three-part study: **Detect** (does the cap exist) → **Reopen** (can interventions undo it) → **Diagnose** (same circuit repaired, or new circuit built).

Hardware: RTX 4060 laptop (8GB VRAM), Ryzen 7 7840. All models train from scratch, no pretrained checkpoints, ever.

---

## CRITICAL PITFALLS — CHECK EVERY BUILD AGAINST THESE

| Pitfall | What It Means | How to Fix It |
|---|---|---|
| **Catastrophic forgetting vs. critical period** | An intervention might "recover" the delayed skill by overwriting other capabilities — the model gets good at arithmetic again but gets worse at general English. That's not real reopening, it's a different tradeoff. | Every reopening evaluation must track **two things simultaneously**: (1) recovery of the delayed skill, (2) retention of general language capability (held-out perplexity), measured before and after each intervention. A "recovery" that tanks general capability should be flagged as forgetting, not reopening. |
| **Pre-training contamination** | If a standard open-source checkpoint (GPT-2, Pythia, etc.) is used, that model has already passed its critical period on standard web data — timing of arithmetic/syntax exposure can't be controlled. | Train every model **entirely from scratch**, from random initialization, using a controlled data mixture that dictates exactly when each skill enters the training stream. No pretrained checkpoints anywhere, including baselines. |

---

## FINALIZED SPECS (locked in — don't re-decide these)

**Skills to test:**
- **Arithmetic:** 2-4 digit addition; carrying vs. no-carrying as a difficulty axis. Synthetic, infinite supply, perfectly gradeable.
- **Syntax:** long-distance wh-question extraction (object-gap constructions) — use BLiMP-style minimal pairs for eval.
  - **MEASURE THE wh-LICENSING INTERACTION, NOT RAW PARADIGM ACCURACY** (settled empirically,
    Week 2 baseline). Raw BLiMP wh_vs_that pairs differ by one word ("what" vs "that"), so the
    score is dominated by the unigram frequency prior for "that": the step-100 baseline scored
    0.000 on wh_vs_that_with_gap and 1.000 on wh_vs_that_no_gap with *zero* syntactic knowledge.
    The interaction (mean of with_gap and no_gap accuracy) cancels the wh-word main effect —
    a purely frequency-driven model lands at exactly 0.500, and only real gap-licensing
    knowledge moves it. Implemented as `syntax.wh_licensing_interaction()`.
  - Week-2 baseline ceiling (5000 steps, 82M tokens): licensing interaction 0.604 local /
    0.534 long-distance. **Long-distance headroom is thin** — if the early-exposure baseline
    barely acquires the skill, a delayed model has little to lose and the Detect phase has no
    gap to find. Week 3 must check this ceiling first (train longer / more skill exposure?)
    before trusting a null on long-distance syntax.
- **Negative control:** word-frequency/vocabulary prediction — should NOT show a critical period; if it does, that flags a problem with the method, not the hypothesis.

**Dataset:** BabyLM Challenge "strict-small" corpus (~10M words) as the natural-language base. Synthetic arithmetic layered on top, injected on a controlled delay schedule.

**Model architecture:** 10M-30M parameters, 4-6 layers, hidden dim 256-512, 8 attention heads, sequence length 256-512 tokens.

**Architecture implementation details (locked in):**
- **Tokenizer:** custom BPE trained on the BabyLM corpus itself, vocab size ~8K-16K (not GPT-2's 50K — wasteful at this corpus size and dilutes embedding-layer capacity)
- **Positional encoding:** RoPE (Rotary Position Embeddings) — handles long-distance dependencies (relevant to the wh-question syntax skill) better than learned absolute positions
- **Normalization:** Pre-norm (RMSNorm or LayerNorm before attention/FFN, not after) — more stable convergence at this depth
- **Activation:** GELU
- **Embedding/output weights: UNTIED** (input embedding and output/unembedding layer are separate weight matrices, not shared). This is a deliberate choice for intervention cleanliness — resetting the embedding layer (intervention #1) should only affect input token representations, not the output prediction layer, so recovery effects can be cleanly attributed to one or the other rather than conflated. Slightly more parameters, negligible cost at this scale.
- **Optimizer:** AdamW, cosine LR decay with warmup (~few hundred steps), weight decay ~0.1 — this is the stable baseline schedule; the LR-spike intervention (#2) is a deliberate deviation from it, so the baseline itself needs to be conventional and well-behaved

**Engineering setup (RTX 4060 / Ryzen 7 7840 optimized):**
- Mixed precision (bfloat16 via `torch.autocast`)
- FlashAttention-2
- Gradient accumulation — micro-batches of 4-8 accumulated before `optimizer.step()`
- `torch.compile()`
- Pre-tokenize the dataset into a `numpy.memmap` binary file; `DataLoader` with `num_workers=8`
- Clean OOP structure: subclass `nn.Module` per attention/FFN block so components can be isolated and reset dynamically mid-training
- Training loop accepts external triggers — manual LR overrides or `requires_grad=False` toggling fired at exact checkpoints
- Log checkpoint evals directly into pandas DataFrames during training; `scipy.stats` for significance testing; `seaborn` for comparative loss curves and heatmaps

---

## STANDING RULE — NO PREMATURE HARDCODING

Checkpoint interval, total training steps, and any delay-onset step numbers derived from them are **provisional until the Week 2 sanity run completes** and gives real total-step counts to tune against.

- Never hardcode these as fixed literals scattered across files (e.g. `if step % 500 == 0`).
- Keep them in one config location (config file, top-of-script constants, or CLI args) so they're a one-line change once real numbers are known.
- Label any placeholder value clearly in code (e.g. `CHECKPOINT_INTERVAL_PLACEHOLDER = 500  # TODO: tune after Week 2 sanity run`).
- This applies broadly, not just to checkpoints: any number that depends on total step count (delay-onset fractions converted to exact steps, intervention trigger points, etc.) should follow the same rule.

---

## STANDING RULE — CONTROL-ARM SEED DEPTH

**Any control arm whose job is to *bound the magnitude* of a confound must be run at the SAME seed depth as the arm it controls — never validated once at a single seed while the primary arm is replicated at n≥3.** A single-seed control cannot distinguish "no effect" from "an effect that happened to land near zero in that one seed"; only per-seed replication shows the confound's magnitude is *stably* null (or stably small) rather than incidentally so. (Applied correctly in the Week-4 seeds-2-3 batch: the syntax negative control was forked per-seed, giving Δ ≈ 0 at every seed, not just seed 1.)

**Companion reporting rule — absolute endpoint over percent-of-gap.** When the baseline/null itself is seed-variable (it is here: d50 arith null ranges 0.506–0.626 across seeds), "recovery as % of the gap" is unstable *by construction* because the denominator moves — a single-seed recovery percentage (e.g. seed-1's "+84%") will not replicate even when the finding is real. Report the **absolute treated endpoint** (± SD across seeds) and the per-seed lift as the headline; treat any percent-of-gap figure as seed-specific and secondary. The stable, replicated claim is "reaches within 90% of the early baseline at n=3," not a fixed recovery percentage.

**Third clause — always show the null endpoint ± SD BESIDE the treated endpoint ± SD, as two numbers, never collapsed into one ratio** (same principle as the retired "~20×" syntax ratio: a single collapsed figure hides which side moved and by how much). "Treated 0.767 ± 0.014" alone doesn't tell a reader whether that's a large or modest recovery — it only becomes interpretable next to "null 0.571 ± 0.061" (here: recovery from a *seed-variable* ~0.57 to a *tight* ~0.77, and note the intervention also collapses the cross-seed variance, itself a finding). And: an endpoint's seed-variance is NOT the same as its *deficit's* seed-variance — computing the latter requires per-seed baselines at BOTH the early and delayed conditions; do not infer deficit-variability from delayed-endpoint-variability alone.

---

## STANDING RULE — CROSS-METHOD VALIDATION REQUIRES DEMONSTRATED DISCRIMINATION

**Never describe a finding as "cross-method validated" until the second method has been shown to
DISCRIMINATE ON THE SAME CONTRAST the first method resolved.** A secondary method agreeing with
the primary one on a single case is not corroboration — it is only corroboration if that method
can be shown to return a *different* answer for the cases the primary method separates. Run the
secondary method across the full set of conditions (including the ones the primary method scored
as negative/opposite) and check its spread and its correlation with the primary readout before
citing it as support.

**Minimum check before claiming corroboration:** (a) the secondary method's dynamic range across
conditions is comparable to the primary's, not compressed near a floor/ceiling; (b) its per-condition
values correlate with the primary readout; (c) it was run on the contrast cases, not just the
headline case.

**This has now cost the project twice** (both Week 8, arithmetic circuit identity):
1. **CKA** — stayed at 0.66–0.82 for *every* triad regardless of whether ablation said repair or
   reorganization; shared init keeps global geometry similar even where causal wiring differs. It
   would have missed the seed-1 rebuild entirely.
2. **Single-head activation patching** — run on seed 1 alone, its "low Jaccard + partial
   sufficiency" result looked like independent confirmation of seed-1 reorganization. Run on all
   six triads it turned out to be flat (ratio 0.59–0.79, SD 0.083, vs ablation transfer 0.19–0.87,
   SD 0.256) with **zero** correlation to the primary readout (r = 0.045, p = 0.93) — it returns
   the same answer for repair and reorganized cases alike. Cause: single-head denoising restoration
   is near-floor for every model (0.05–0.15), so the ratio divides two tiny numbers and has no
   resolving power. The "reorganization-with-redundancy" reconciliation built on it was retracted.

Corollary: when only ONE probe discriminates (here mean-ablation — whole-layer ablation saturates,
CKA and single-head patching are flat), say so explicitly and report the finding as single-method.
That is an honest result; a fake second method is worse than none.

---

---

## STANDING RULE — ARCHIVE, NEVER DELETE

**Never delete final (step-5000) or fork-point checkpoints, eval logs, or any artifact of
a run whose results appear in the paper. Intermediate weight checkpoints of completed runs
are prunable, since training is deterministic and resume-validated, making them
regenerable.**

This documents what `reopen/prune.py` already enforces (onset-fraction + final keep-set,
incomplete runs never pruned, dry-run by default) rather than changing it. The companion
convention for analysis outputs is `reopen/incremental.py:archive()` — a timestamped `.bak`
rename, never an `unlink`.

---

## STANDING RULE — AUDIT THE LEDGER AGAINST THE ORIGINAL GOALS, NOT ONLY AGAINST ITSELF

**Periodically check the claim ledger against the master plan's stated goals and the
pre-registered hypotheses directly. A ledger can be perfectly self-consistent and still
be silently missing a finding the project existed to produce.** Internal consistency
checks — every claim labelled, labels matching across abstract/section/discussion, no
banned phrasings — cannot detect an *absent* row, because there is nothing inconsistent
about a claim that was never written down.

**This happened on 2026-08-03.** The arithmetic-vs-syntax dissociation had no ledger
entry, though the master plan names "a clean dissociation" as hit-criterion #1. Claims 1
(arith CP) and 11 (syntax null) each existed; nothing conjoined them. Because anchor
options for the write-up were generated *from the ledger*, the finding was invisible to
every one of them — an omission propagates silently in a way an error does not, since
there is no wrong text for a reviewer to catch. Added as claim 11b.

**How to run the audit:** walk the master plan's goals and the pre-registered predictions
one at a time and ask "which ledger row answers this?" — not the reverse direction, which
is what self-consistency already covers. Expect three outcomes, and record all of them:
(a) a row answers it; (b) no row answers it and one should exist → write it, with a label;
(c) the project answered a *different question* than the goal stated → say so explicitly
rather than letting the near-miss pass as a hit. Case (c) is the easiest to miss: the
dissociation found was *detection*-side (arith has a CP, syntax doesn't) while the plan
anticipated a *reopening*-side one (arith is reopenable, syntax isn't) — superficially the
same finding, not actually the same claim.

Companion to the §7 banned-phrasings re-audit rule in `notes/week9-stats-backbone.md`
(audit the *approved replacements*, not just the banned entries): both exist because a
trusted artifact distributes whatever it contains, including absences.

---

## CURRENT BUILD PRIORITIES

**Checkpoint convention (locked in — needed for delay/intervention precision):**
- Checkpoint on a **fixed step interval**, not epoch boundaries or wall-clock time — e.g. every 500 or 1000 training steps (pick based on total planned steps so you get 40-100 checkpoints per run; exact number to tune once total step count is known from Week 2 sanity run)
- **Total training steps must be decided upfront per run**, since delay onsets (0%/25%/50%/75%) and intervention trigger points are defined as *fractions of total steps* — the scheduler needs to convert e.g. "delay to 50%" into an exact step number before training starts, not compute it on the fly
- **Naming convention:** `checkpoints/{run_id}/step_{step_number:07d}.pt`, where `run_id` encodes the run's config, e.g. `arith_delay50_size20M_seed1` — this must be readable by both the eval harness (to log results against the right run) and the intervention module (to know which checkpoint to resume from and apply an intervention to)
- Each checkpoint file should store, alongside model weights: current step number, optimizer state, and a small metadata dict (skill, delay_onset fraction, model_size, seed, and — once relevant — which intervention was applied at which step). This metadata is what makes the Week 9 statistical analysis possible without re-deriving run configs from filenames
- Delay-scheduler triggers and intervention triggers fire based on **exact step count**, never wall-clock time, so results are reproducible run-to-run regardless of GPU thermal throttling or background load

**Build first:**
1. Basic nanoGPT-style model + training loop, random-init only, running on the BabyLM corpus alone (sanity check — confirm loss decreases sensibly)
2. Synthetic arithmetic data generator + syntax-filtered data extraction from BabyLM/BLiMP
3. Delay scheduler wired into the data pipeline
4. Eval harness logging **both** skill accuracy and general perplexity at every checkpoint from day one — this pairing must exist from the start, not bolted on later

**Hold off on for now:**
- Intervention module (reset/LR-spike/freeze/cram) — not needed until the Week 3 pilot confirms a critical period exists
- Circuit diagnosis tooling (activation patching, ablation, pruning) — needed only once recovery is observed, later in the study

---

## WEEK-BY-WEEK WALKTHROUGH (so build order matches research phase)

### Week 1 — Literature + Hypothesis (DONE)
Papers read, skills/dataset/architecture confirmed, predictions finalized (see below).

### Week 2 — Infrastructure (CURRENT)
- Build nanoGPT-style training pipeline, random-initialization-only
- Build data generators: synthetic arithmetic, syntax-filtered corpus from BabyLM, delay scheduler
- Build the intervention module structure (reset/LR-spike/freeze/cram) — scaffold now, doesn't need to be wired into the training loop yet
- Build the eval harness logging both skill accuracy and general perplexity at every checkpoint
- Sanity-check with one full baseline run

### Week 3 — Pilot: Detect
- Small grid: 1 model size × 3 delay onsets × 2 skills
- Confirm total-exposure is matched across conditions (only timing differs)
- Check for any delay-dependent performance gap — validates the method before scaling up
- **EXIT GATE (go/no-go before Weeks 5-7 grid): vocab-control signal check.**
  Run `cloze_scores(..., per_word=True)` on pilot checkpoints (original eval set) and decide:
  - Clean (low cross-word variance, smooth aggregate log-prob) → proceed to grid.
  - Noisy but aggregate stable → bump `n_eval` / `max_eval_per_word` (cheap: regenerates
    eval set + vocab.bin, no pilot rerun needed) → re-check, then launch grid.
  - Still underpowered → widen the frequency band (expensive: full vocab scrub rebuild
    + rerun of pilot vocab conditions) — do it HERE, never mid-grid.
  Any change regenerates vocab.bin, so all grid runs must train on the final version;
  eval numbers are only comparable between models trained on the same vocab.bin.

### Week 4 — Pilot: Reopen
- Take delayed-exposure models from Week 3 that show a gap
- Apply each intervention type at a fixed point, continue training with equal budget
- Check for recovery on the target skill **and** confirm general perplexity didn't degrade
- **COMMON-CHECKPOINT RULE:** every intervention run resumes from the IDENTICAL saved
  checkpoint file (same weights, same optimizer state, same step) — never from
  independently-continued runs that drifted apart. Recovery percentages are only
  comparable across interventions if the starting point is bit-identical. The
  post-resume data stream must also be identical across the four runs (same seed →
  same reconstructed stream), except where the intervention itself changes it (cram).
  NOTE: the Trainer needs a resume-from-checkpoint path built for this — the
  checkpoint files already store everything required, the loading side is the
  one remaining piece of wiring.
- **RESET × LR-SCHEDULE CONFOUND:** a reset fired late in training re-randomizes
  weights under a cosine-decayed LR — fresh embeddings may get too little gradient
  signal to relearn, making reset look like a failed intervention when it's an LR
  artifact. But don't just silently bolt an LR bump onto reset: LR decay is itself a
  candidate mechanism of critical-period closure, so the pilot should run BOTH
  reset-alone AND reset+LR-spike (or a local-LR-warmup-for-reset-params variant) as
  separate conditions — the contrast separates representation entrenchment from
  optimizer-schedule plasticity. Resolve in the Week 4 pilot, not mid-grid.

### Weeks 5-7 — Full Experiment Suite
- Detect: 3 model sizes × 4-5 delays × 2-3 skills × 3 seeds
- Reopen: for skill/delay combos showing a real gap, run all 4 intervention types × 3 seeds
- Batch overnight; checkpoint review after week 5-6

### Week 8 — Circuit Diagnosis
Build activation patching, weight/representation similarity (CKA/cosine), ablation transfer test, pruning-based localization — only needed once recovery is actually observed in Weeks 5-7.

### Week 9 — Statistical Analysis (CURRENT)
**Backbone decided: `notes/week9-stats-backbone.md`; numbers computed by
`scripts/week9_stats_backbone.py` → `logs/week9_stats/`.** All three designs originally
planned here are **RETIRED as un-estimable on the surviving design** — the write-up states
what was planned and why it was not run:
- ~~Detect: mixed-effects, final accuracy ~ delay × model_size~~ → after the pre-registered
  11M FLOOR exclusion the design is 12 observations with one per seed×cell; no within-cell
  replication for a random effect. **Replaced by** per-seed *paired* deficit per size,
  Holm across the two confirmatory sizes, with the MDE quoted for the size-moderation null.
- ~~Reopen: recovery percentage ~ intervention_type × skill × delay_onset~~ → recovery-%
  denominator is the seed-variable null (already banned); `skill` and `delay_onset` are
  constants; only one intervention has n=3. **Replaced by** the pre-registered absolute-
  endpoint gate + per-seed paired lift, with n=1 arms scored against the null's cross-seed
  *range*.
- ~~Diagnose: circuit-overlap ~ intervention_type × skill~~ → both factors are constants.
  **Replaced by** deficit-ordered ablation transfer (exact-permutation Spearman + LOO),
  reported as single-method by demonstration.

Every claim carries one evidential label (ESTABLISHED / SUGGESTIVE / BOUNDED-NOT-EXCLUDED)
fixed in the backbone's claim ledger, plus a NOT-A-CLAIM list for computed-but-inert
numbers. Labels must match across abstract, section, and discussion.

### Weeks 10-13 — Writing, Polish, Submit
No code needed here — human-facing work.

---

## OPERATIONAL DEFINITIONS (needed for eval harness logic)

**Critical Period (Detect):** Skill S has a critical period ending at checkpoint T if a model first exposed to S after T reaches significantly lower final accuracy than a model exposed from the start — even with matched total S-token exposure — and the gap persists after extended fine-tuning on S alone.

**Reopening:** An intervention successfully reopens S's critical period if it brings the delayed model's accuracy to within a set margin (e.g. 90%) of the early-exposure baseline, *without* degrading general-language perplexity beyond a matched threshold. Recovery at the cost of general capability counts as forgetting, not reopening.

**Circuit Identity (Diagnose):** A recovered model *repaired the original circuit* if activation patching at the baseline's relevant layers/heads reproduces its skill behavior, and ablating those components damages performance similarly to the baseline. It *built a new circuit* if patching/ablating the baseline's components has little effect on the recovered model.

---

## PREDICTIONS (context for why certain interventions get closer monitoring)

1. **Syntax is predicted to show the stronger critical period** — long-distance dependencies (wh-question object-gap extraction) likely rely on distributed representations scaffolded by earlier-learned structure, more fragile to a late start than arithmetic.
2. **Embedding/layer reset predicted as most effective intervention** — but also the biggest forgetting risk, since it wipes existing representations. This is why the eval harness must log general perplexity at every single checkpoint, not just for reset but for all four interventions.
3. Circuit identity outcome — open, but linked to #2: if reset wins, expect "new circuit" more often in Week 8 diagnosis, since reset destroys the original representation entirely.

## THE 4 REOPENING METHODS (Tier 1 — implement these)

1. **Embedding/layer reset** — reset embedding (or specific layers) to fresh random init partway through training, then continue. Biggest forgetting risk, monitor general perplexity closely.
2. **Learning rate spike** — temporarily boost LR right before/during re-exposure, then decay back down.
3. **Layer freeze/unfreeze** — freeze most of the network, unfreeze only specific layers during re-exposure.
4. **Concentrated re-exposure ("cram phase")** — feed a burst of unusually high-density skill data for a short stretch. Control condition.

## TIER 2 FALLBACK METHODS (only implement if all 4 above show zero recovery)

- Optimizer state reset (Adam moment estimates only, not weights)
- Cyclical LR restarts (repeated cosine-annealing restarts)
- Adapter/LoRA modules (small trainable module dedicated to the missed skill)
- Net2Net-style network growth (expand network at intervention point)
- Lottery-ticket rewind (rewind subnetwork to earlier checkpoint, retrain just that subnetwork)
- Spaced repetition (interleave missed skill at varying intervals)
- Replay of early data + skill (reintroduce early corpus slice alongside new skill)
- Distillation from a "healthy" twin (early-exposure model's behavior distilled into delayed model)

## CIRCUIT DIAGNOSIS TOOLING NEEDED (Week 8, build when reopening confirmed)

- Activation patching / causal tracing
- Weight/representation similarity (CKA or cosine similarity)
- Ablation transfer test
- Pruning-based localization

## STATISTICAL ANALYSIS — SUPERSEDED

The three designs formerly listed here are retired; see the Week 9 entry above and
`notes/week9-stats-backbone.md` for the tests that replace them, the whole-paper claim
ledger, the multiplicity families, and the number→owner map.
