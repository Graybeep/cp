# Two versions, no findings dropped — compiled

Source: `scripts/two_versions.py`. Every number is a pdfTeX page count. **No venue limit is
applied or targeted here** — the point is to find out what shape the paper is, and choosing a
venue from the number is the next step rather than this one.

## The two builds

| | ACL two-column, body | + appendix | NeurIPS, body |
|:--|--:|--:|--:|
| **V1 Full** — everything standalone, nothing moved | **12 pp** | 14 pp | 14 pp |
| **V2 Condensed** — merged figure (modelled) | **12 pp** | 14 pp | 13 pp |
| V2 Condensed — two figures, unmerged | 12 pp | 14 pp | — |

**V2 costs 0 pages less than V1 with the merged figure, and 0 without it** — and no claim, section,
table or figure differs between them. Only apparatus moved.

## What V2 actually contains

Identical to V1 in every result. The differences, all measured claims-neutral in
`logs/move_cost.md` before being applied here:

- **§2** — engineering settings, the embedding-share arithmetic and the three un-estimability
  derivations move to the appendix. The cosine-decayed LR schedule and untied embeddings stay
  in the body deliberately: claim 7 says closure *is* that schedule, and the reset intervention's
  validity rests on the embedding/head separation.
- **§6** — claim 21's power numbers move; the claim, its direction and its *suggestive* label stay,
  because a labelled claim's qualifiers may not outrun it into supplementary material.
- **§8** — the 14 / 59 / 82 seed-count derivations move; each limitation keeps its named experiment.
- **Related Work** — demoted to a Discussion paragraph keeping the plasticity/warm-starting
  positioning at near-full length. **The demoted paragraph needed a clause adding back**: the drafted
  version (`cut_dependencies.RW_PARAGRAPH`) was written for the plan that *deleted* §5 and so drops
  the circuit-analysis citations. §5 stays here, so they return (`CIRCUIT_CLAUSE`).
- **§5 and §6 stay in the body, uncompressed.** See the caveat below — this is the one part of the
  brief this file could not execute.

## Appendix C — written, so the moves now land

The §2, claim-21 and §8 moves do not delete their material, they **point at it**, and their
stubs refer the reader to *Appendix C* five times. That appendix did not exist until now:
`notes/draft-appendix.md` held only A and B, so the condensed build promised material it did
not carry. **An appendix move that never lands is how content leaves a paper while every page
count still looks correct** — the length machinery cannot detect it, because a dangling pointer
costs no pages.

Appendix C now carries all five: architecture and optimizer settings (C.1), parameter counts
and the embedding share (C.2), the three un-estimability derivations (C.3), the attraction
trajectory's power calculation (C.4), and the outstanding-test seed counts (C.5).

**One stub was over-promising and was reworded rather than filled.** It said "Appendix C gives
the per-seed values" for claim 21's trajectory — but the step-2500 evaluations were never
retained per seed, only the paired aggregate, so those values exist nowhere in the logs. It now
promises the power calculation, which does exist. The alternative was inventing six numbers.

**Appendix C belongs to V2 only.** V1 moves nothing and carries that material inline, so it
takes Appendices A and B; giving it C would print the same content twice.

## Three things not done, and why

1. **The pruning pre-screen and the classifier provenance were already moved** (2026-08-07,
   `notes/draft-appendix.md` A and B). §5's body already carries only the one-sentence versions —
   pruning named as the third non-discriminating probe with its reason, and the unreachable-label
   fact. There is nothing left to move, and that move is already inside every number above.
2. **No measured compression of §5 or §6 exists.** 1.85 and 0.65 are budget targets from the
   2026-08-07 rebalance, not edits that were ever applied and re-wrapped; §5 currently measures
   2.13 estimated pages against that 1.00 budget and §6 0.90 against 0.65. Compressing them is
   real work with a real risk — both are dense with labelled claims whose qualifiers the label
   rule forbids dropping — so it is not silently included in a number here. **If V2 needs to be
   smaller, this is the next move and it has to be costed with `move_cost.py` first.**
3. **The merged figure is not drawn** — and at this length it does not matter. Its slot cost is
   modelled (30 slots at 7.1 × 2.80 in), so V2 was compiled both ways, and the two rows are
   **identical at 12 pp**. Merging bought a page in the
   section-drop plan only because that plan sat one line the wrong side of a page boundary;
   V2 is nowhere near one. **So the C2 drawing is no longer on the critical path** — it should be
   done for the reader (it removes real duplication) rather than for the page count.

## Reading the number

**V2's body is 12 pages, and the whole condensation bought 0.** That is the finding, and it is
worth stating as a negative: every apparatus move available — three §2 moves, claim 21's power
numbers, §8's derivations, a demoted Related Work, two appendices already banked, and a merged
figure — comes to 0 page between them. **The paper's length is its results, not its packaging.**
That is the honest reading of every length exercise in this project converging on the same
answer from a fourth direction.

At 12 pages it is not workshop-shaped on any of the three CFPs read (limits 5, 8 and 9), and
not by a margin any packaging decision reaches. The three options left are a venue whose limit
fits this shape, an explicit length exception, or compressing §5/§6 — and only the last is still
a *drafting* decision rather than a submission one. It is also **the only one that risks the claims**,
which is why item 2 asks for it to be costed before it is attempted rather than after.

V1 is unaffected by all of it: 12 pp of body, 14 with the appendices. Note that every
number in this file still **excludes references**, for comparability with every other length
measurement here. The actual arXiv package adds them — see `logs/arxiv_build.md`, which is
the build that goes out.
