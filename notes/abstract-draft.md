# Abstract — draft 1 (2026-08-04)

Anchors per `week9-stats-backbone.md` §12: claims **1 → 11b → 4 → 7**.
Every sentence is mapped to a claim and label in the audit table below.

---

## Draft

> Critical periods (windows after which a skill becomes hard to acquire regardless of
> further data) are usually studied in models whose pretraining history cannot be
> controlled. We train transformers of 11M–60M parameters from random initialization on a
> ~10M-word corpus, injecting synthetic arithmetic and long-distance *wh*-licensing on
> delay schedules that hold total skill exposure fixed and vary only *when* that exposure
> begins.
>
> Delaying arithmetic exposure to the midpoint of training produces a persistent
> endpoint deficit against seed-matched early-exposure anchors: 0.214 ± 0.055 at 23M and
> 0.229 ± 0.037 at 60M, in the same direction at every seed. Long-distance *wh*-licensing
> shows no detectable deficit under five distinct probes, inverting our pre-registered
> prediction that syntax would show the *stronger* critical period. We claim this
> dissociation in absolute terms and not as differential susceptibility, because the two
> skills are not equally observable: the syntax anchor sits 0.028 above its metric's
> chance floor against arithmetic's 0.791, so a proportionally matched deficit would have
> fallen below the design's resolution (MDE 0.048). Four of the five probes ask whether a
> critical period exists *given the acquisition that occurred*; only a representational
> probe, which carries independent dynamic range (selectivity 0.60, p < 1e-8) and is flat
> across timing (p = 0.48): asks whether enough was acquired for one to have been
> detectable at all. That probe, not the count of nulls, is what the dissociation rests on.
>
> Where the deficit exists, it reopens. Shifting the learning-rate schedule at the delay
> onset restores endpoint accuracy to 0.767 ± 0.014 at 23M and 0.796 ± 0.022 at 60M,
> against untreated nulls of 0.577 ± 0.051 and 0.600 ± 0.040: at 23M with no measurable
> cost to held-out perplexity, at 60M with a cost bounded above by 1.67 ppl that is
> plausibly an annealing-horizon artifact. Both recovered endpoints fall inside a
> pre-registered band, so recovery converges on a scale-invariant level rather than
> tracking capacity, and it collapses the cross-seed variance the untreated deficit shows.
> Concentrated re-exposure to the missed skill does not reproduce this: a data-only cram
> phase stays near the null, while cram combined with the schedule shift is
> indistinguishable from the shift alone. Closure at this onset is therefore largely a
> property of the optimizer's decayed learning-rate schedule rather than of entrenched
> representations.
>
> The two results are timing effects at different levels: *which* skills a late start
> damages is skill-specific, while the damage itself is carried by the optimizer's
> schedule rather than by representations that have hardened against revision.

---

## Sentence → claim audit

| sentence | claim | label | obligation honoured |
|:--|:--|:--|:--|
| setup, matched exposure | — | — | timing-not-dose framing; "vary only when" is what licenses a timing claim |
| deficit 0.214/0.229, every seed | **1** | ESTABLISHED | "produces" permitted; per-seed direction stated |
| LD null, five probes, inversion | **11**, pred #1 | ESTABLISHED (null, scoped) | §11 requires the inversion in-abstract |
| "absolute terms, not differential susceptibility" + headroom + MDE | **11b** | ESTABLISHED **(scoped)** | §12: scoping appears *in* the abstract, not deferred |
| four-vs-one probe split; probe 4 load-bearing | **12** | ESTABLISHED | the two-jobs distinction, un-blurred |
| recovery endpoints beside nulls | **4** | ESTABLISHED | endpoints paired, never percent-of-gap |
| ppl: 23M none / 60M ≤1.67 bounded | **10** | BOUNDED-NOT-EXCLUDED | quoted *with* its bound; forgetting-vs-reopening pitfall |
| band → scale-invariant; variance collapse | **5**, **6** | ESTABLISHED | claim 5 kept as non-anchor per §12 |
| cram ≈ null; cram+shift ≈ shift | **7** | ESTABLISHED | mechanism stated as the consequence, not asserted bare |
| linking sentence | 11b × 7 | — | §12 required link; without it the two anchors read as two papers |

## Deliberately absent

- **Predictions #2 and #3** — §11 locates both in the Discussion, not the abstract.
  Do not let "a data-only cram phase stays near the null" drift into a reset claim.
- **The priced 60M rate-collapse test** (14 delayed seeds) — belongs in Limitations,
  phrased as the specific experiment that would resolve 11b's susceptibility half.
  It must not soften into "future work could investigate".
- **The 11M floor exclusion** — reported in §2.1 and §8. Add a parenthetical here only
  if a reviewer reads "11M–60M" and then finds no 11M confirmatory numbers.

## Cut priority if over length

1. "and it collapses the cross-seed variance…" (claim 6 — real, but the least
   load-bearing of the C2 group)
2. "in the same direction at every seed" (recoverable from the ± SDs)
3. The parenthetical statistics inside the probe-4 clause — but **not** the clause itself;
   losing it reintroduces the "five converging nulls" overclaim
4. Never cut: the headroom contrast (0.028 vs 0.791), the 60M ppl bound, or the
   absolute-vs-susceptibility distinction. Each is a label obligation, not a nicety.

**~330 words.** If the venue caps at 250, apply cuts 1–3 and compress the setup sentence.
