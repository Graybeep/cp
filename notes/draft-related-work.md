# Draft — Related Work (draft 2, 2026-08-07)

Placed between §5 and §6 per the applied rebalance; budget 0.45 page (≈ 350 words), measured
at 511 (see the length note at the end). Draft 2 distinguishes grokking explicitly and
compresses the development and circuit-identity citations.

> **⚠ CITATIONS ARE UNVERIFIED.** The references below are named from background knowledge,
> not checked against a bibliography or the actual papers, and my knowledge has a cutoff.
> Every one needs verifying before submission — author, year, venue, and above all *that the
> paper says what I have it saying*. Treat the argument structure as the deliverable and the
> citation keys as slots. Two in particular carry weight and must be read, not cited from
> memory: the warm-starting and loss-of-plasticity lines, because they are the strongest
> candidate for "this is a rediscovery".

---

# Related Work

**Critical periods.** The concept comes from development, where early sensory deprivation
produces deficits that later normal input does not repair [Hubel & Wiesel; Lenneberg;
Johnson & Newport]. The closest machine-learning analogue is Achille et al.'s demonstration
that deep networks have critical learning periods: a deficit imposed early in training
(blurred images) leaves a permanent gap even given unlimited clean data afterwards, and the
window's timing is a property of training rather than of architecture [Achille, Rovere &
Soatto, ICLR 2019]. That work manipulates *input quality* in vision models; we manipulate
*when a skill's data arrives* in a language model while holding its total quantity fixed,
which is what lets a deficit be attributed to timing rather than to dose or degraded signal.

**Plasticity loss and warm starting.** A separate literature finds that networks become
harder to train as training proceeds: warm-started models generalize worse than freshly
initialized ones given the same final data [Ash & Adams, NeurIPS 2020], and continual-learning
agents progressively lose the ability to fit new tasks, recoverable by re-injecting
plasticity through resets or noise [Dohare et al., Nature 2024; Lyle et al.; Berariu et al.].
**This is the closest prior explanation of our reopening result and the one we most risk
rediscovering**, so we state the difference precisely: those settings involve a distribution
shift, a restart, or a sequence of distinct tasks, and typically recover plasticity by
perturbing *weights*. Our models train once, without interruption, on a stationary mixture
whose only manipulated variable is a skill's onset step, and the intervention that recovers
the skill changes no weights at all, only the learning-rate schedule, while resetting weights
(the standard plasticity remedy) produced no detectable recovery in our setting at *n* = 1.
The mechanisms may still be the same one seen from two directions; what we can say is that
the recovery here is obtainable without weight perturbation. (That reset arm is *n* = 1;
§7.2 states what it does and does not license.)

**Curriculum, data ordering, and delayed generalization.** Ordering effects have a long
history [Bengio et al., 2009] with mixed results at LM scale, where reordering often yields
little; our finding is compatible with that, since total exposure is what matters for most
of the corpus and it is one late-arriving skill that pays a timing cost. Delayed
generalization on arithmetic is also studied as grokking [Power et al., 2022], which is a
different phenomenon despite the surface adjacency: grokking is a long plateau followed by
sudden generalization at *fixed* data, whereas ours is a permanent endpoint gap produced by
*shifting* when the data arrives.

**Small-scale LM evaluation.** We use the BabyLM setting [Warstadt et al., 2023] and
BLiMP-style minimal pairs [Warstadt et al., 2020], with the metric modification of §2.3, and
the agreement-attraction literature [Linzen, Dupoux & Goldberg, 2016; Marvin & Linzen, 2018]
for §6.

**Circuit identity.** Our diagnosis uses causal ablation and activation patching in the
circuit-analysis tradition [Elhage et al., 2021; Wang et al., 2022; Meng et al., 2022] and
representational similarity [Kornblith et al., 2019]; §5.1 reports the negative
methodological result that three of those four probes do not discriminate on our contrast.

---

## Notes for the next pass

- **Grokking — RESOLVED 2026-08-07, distinguished explicitly** in the curriculum paragraph.
  A reviewer working on grokking would otherwise ask, and the distinction is one clause: the
  data is fixed there and the *onset* is what moves here. It is a distinction, not a
  contrast — we are not claiming the two are unrelated, only that our result is not an
  instance of theirs.
- **The reset asymmetry is in the plasticity paragraph and is the substantive part** of the
  contrast: weight resetting is the standard plasticity remedy and it was inert here, while
  the schedule-only intervention worked. It stays at "no detectable recovery" because it is
  *n* = 1 (claim 9, SUGGESTIVE) — the sentence must not harden into "reset fails".
- **Length — measured, not estimated (2026-08-07).** **511 words against a 350-word budget**
  (0.64 page against 0.45; `logs/length_audit.md`). Note the direction: the two trims made
  today (the development citation list, the circuit-identity paragraph) saved ~35 words and
  the grokking distinction cost ~75, so the section got *longer* on purpose — the
  distinction is worth more than the words. The compressible material is now compressed.
  What remains over budget is the plasticity paragraph, which is the section's
  load-bearing claim and is where a reader decides whether the schedule finding is a
  rediscovery — **do not pay the paper's 2.3-page overflow out of this section.** §5 and
  §7–§9 are 70 % of that overflow between them.
- Still open: whether the small-scale-LM paragraph's citations belong here at all or belong
  inline in §2, which would save ~45 words and is a formatting call, not a content one.
