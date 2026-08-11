# LR-budget schedule-shape test for LD (user idea #2)

**Question:** arithmetic's critical period turned out to be an LR-schedule problem (cosine
decaying too low late in training; the lr_shift intervention rescued it 0.506→0.752). Is
long-distance syntax's floor the same kind of problem — does the standard schedule simply
not give LD enough *sustained high-LR* training to consolidate, independent of timing?

**Design:** delay-0 syntax baseline, everything matched (23M, seed 1, 5000 steps, exp 0.1,
warmup 300, peak lr 6e-4, exposure = 8.192M syntax tokens), the ONLY change being the cosine
floor `min_lr_ratio` 0.1 → 0.5 (floor 6e-5 → 3e-4). LR stays high through training instead
of decaying to near-zero — NOT more steps (total_steps fixed; matched exposure preserved).
run_id `syntax_delay0_exp10_size23M_seed1_hilr50`. n=1 exploratory.

**Result (continuation metric, primary):**

| metric | baseline (min_lr 0.1) | hilr50 (min_lr 0.5) | Δ |
|---|---|---|---|
| LD interaction | 0.586 | 0.603 | +0.017 |
| local interaction | 0.679 | 0.700 | +0.021 |
| with_gap_LD (hard dir) | 0.303 | 0.344 | +0.041 |
| no_gap_LD | 0.868 | 0.863 | −0.005 |
| val_ppl | 46.15 | 45.68 | −0.47 (better) |

**Read: NOT an LD-specific rescue — the schedule mildly helps everything.**
- LD moved +0.017 — about **1 seed-SD** (the 24-seed delay-0 continuation-LD SD is 0.0157).
  This is a within-noise nudge, and seed 1 is already the high-acquisition outlier (its
  standard-schedule 0.586 is near the top of the 24-seed distribution).
- Critically the lift is **non-specific**: local moved +0.021 (as much as LD) and val_ppl also
  improved slightly. A hotter sustained schedule gave a small uniform benefit to the whole LM,
  not a targeted unlocking of the LD dependency. Contrast arithmetic, where lr_shift moved the
  target skill by +0.25 while being skill-specific.
- No forgetting (val_ppl improved), so the manipulation is benign — but benign-and-non-specific
  is exactly "schedule shape is not the LD bottleneck."

**Conclusion:** the LR-budget mechanism that dominated arithmetic's critical period does **not**
open LD headroom. This rules out — in this exact setup — the one closure mechanism already
proven to matter, which is precisely the informative negative idea #2 was designed to get.

**Pre-committed caveat:** n=1 on the seed-variable, high-outlier seed 1. The only residual
ambiguity is whether the +0.017 is stable noise or a small *real but non-specific* effect;
hilr50 seeds 2-3 (~1 hr) would settle that. But since the lift is non-LD-specific either way,
replication cannot convert this into a critical-period rescue — the qualitative conclusion
(schedule shape is not the LD bottleneck) does not depend on it.

## Replication (n=3, ran 2026-07-25) — CONFIRMS non-specific, LR-budget RULED OUT

| seed | baseline LD | hilr50 LD | Δ LD | Δ local | val_ppl |
|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.586 | 0.603 | +0.017 | +0.021 | better |
| 2 | 0.522 | 0.532 | +0.010 | +0.015 | better |
| 3 | 0.520 | 0.527 | +0.007 | +0.022 | better |
| **mean** | | | **+0.011 ± 0.005** | **+0.019 ± 0.004** | all improve |

- Δ LD (+0.011) is within the baseline seed-SD (0.0157) — within noise.
- **local moves MORE than LD on average (+0.019 > +0.011), every seed** → the sustained-high-LR
  schedule helps the whole LM slightly and uniformly; it is the *opposite* of an LD-specific
  rescue. val_ppl improves in all three seeds (benign schedule, no forgetting).
- The seed-1 +0.017 does not replicate as an LD effect. **LR-budget mechanism ruled out for LD,
  confirmed n=3.** Fifth independent null for the syntax-LD critical-period question, replicated.
