# What each proposed move actually buys

Source: `scripts/move_cost.py`, measured with `page_estimate.py`'s own wrap on the
geometry taken from the compile (219.1 pt column, 10.95 pt on 13.6 pt, 102 lines/page).
Every figure below is a re-wrap of the edited draft, not a word count divided by anything.

| move | section now | words out | words back | **cut** (delete outright) | **move** (delete, pay the stub back) |
|:--|--:|--:|--:|--:|--:|
| A. Claim 21 (§6 attraction-over-training) leaves the body | 92 ln | 89 | 47 | −11 ln (0.108 pg) | −5 ln (0.049 pg) |
| B. §8 seed-count derivations (14 / 59 / 82) to the appendix | 141 ln | 34 | 9 | −4 ln (0.039 pg) | −3 ln (0.029 pg) |
| C. §2.1 engineering settings to the appendix (LR schedule + untying kept) | 179 ln | 177 | 84 | −24 ln (0.235 pg) | −12 ln (0.118 pg) |
| D. §2.1 embedding-share arithmetic to the appendix (caveat kept) | 179 ln | 104 | 50 | −15 ln (0.147 pg) | −7 ln (0.069 pg) |
| E. §2.6 un-estimability derivations to the appendix | 179 ln | 50 | 22 | −6 ln (0.059 pg) | −3 ln (0.029 pg) |
| **C+D+E together: the whole §2 appendix move** | 179 ln | 331 | 156 | −45 ln (0.441 pg) | −22 ln (0.216 pg) |
| **all distinct moves (A–E)** | | 454 | 212 | **−60 ln (0.59 pg)** | **−30 ln (0.29 pg)** |

Against the measured overflow of **+465 lines (+4.56 page)**:
- as appendix *moves*, all of them together close **6 %** of it;
- as outright *cuts*, **13 %**.

## §2 is where the appendix finally works

C+D+E buy more as an appendix **move** (23 lines) than claim 21 and the §8 derivations
buy as outright **cuts** (15). That is the predicted asymmetry, and it is the first time
the appendix has paid: §5 and §6 are dense with labelled claims, which must be restated
in the body and so come back as a stub, whereas architecture constants, an embedding-share
calculation and three un-estimability derivations are what supplementary material is for.

Two items were held out of C deliberately, and both for reasons that come from results
elsewhere rather than from taste: the **cosine-decayed LR schedule** (claim 7 says closure
at d50 *is* that schedule, so a reader cannot check the mechanism argument with it in an
appendix) and **untied embeddings** (the reset intervention's validity rests on the input
representation and the prediction head being separable).

## What size of move would work

These edits also calibrate the exchange rate, which is the number that should size every later move: **454 words of running prose bought 60 lines = 7.6 words per line.** (The whole-body average is 6.5, but that counts headings, tables and figure slots, which hold lines without holding words. For deciding what a *paragraph* cut buys, the marginal rate above is the right one.)

So closing +465 lines out of prose costs on the order of **3,519 words** — against a body of **8,006 words**, i.e. about **44 % of everything written**. Every move costed here removes 454 words between them, 13 % of that. No collection of paragraph-level moves reaches it; the deficit is structural, and the remaining options are whole sections and the page limit itself (`logs/section_menu.md`).
