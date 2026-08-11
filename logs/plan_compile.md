# The leading plan, compiled — and what the figure consolidation is worth

Source: `scripts/plan_compile.py`. Every number below is a pdfTeX page count of the
leading plan's actual text: §5 and §6 omitted, Related Work demoted to `cut_dependencies.RW_PARAGRAPH`
inside the Discussion, `move_cost` rows C–E applied to §2, and §5's dependents resolved
as `cut_dependencies.DEPS` specifies. The edits are imported from the scripts that costed
them, so the compiled paper and the costed plan cannot diverge.

## 1. The plan compiles to a real page count

| | ACL (limit 8) | NeurIPS (limit 9) |
|:--|--:|--:|
| current drafts, everything | 13 pp (+5) | — |
| **leading plan** | **9 pp (+1)** | **10 pp (+1)** |

The estimate this replaces was **8.46 pages plus a carried 0.48-page offset ≈ 8.9**, and it
was right: the compile says 9. That is worth stating plainly rather than quietly
dropping, because the reasoning behind the offset was still unsound and would not have been
self-correcting. The 0.48 was the estimator's residual on the *full* draft, carried across as
though it were a property of the estimator; it is a property of the text — float placement and
multi-line table cells — and the leading plan deletes §5 and §6, which hold a large share of
both. An extrapolation that cannot be checked is not made safe by landing in the right place.

What the compile adds is not a different number but the two things the estimate could not
give: the paper is over by a **whole page, not 0.9 of one** — page counts are integers and a
0.1-page distinction between them was never real — and the figure allowance underneath it is
wrong in a way that changes the decision (§2, §3 below).

## 2. The figure allowance was never measured

`FIGURE_LINES` charges **38 line-slots per figure** and its own comment calls that "~0.30 page each" — a budget written before the figures existed. The figures are 7.1 × 2.45 in and 7.1 × 2.7 in (`make_figures.py`), and the text block
measures **455.2 pt** wide (read off the compile, not derived). Both are wider than the text
block, so both scale down to it and their aspect ratio fixes their height:

- figure 1 costs **27 slots**, not 38;
- figure 2 costs **29 slots**, not 38;
- together **57 slots against the 76 assumed** — the budget over-charges the figures by
  19 slots (0.19 page).

## 3. The consolidation, priced as a curve

| figure treatment | slots | compiled |
|:--|--:|--:|
| 0 — no figures at all (floor, not a proposal) | 0 | 8 pp |
| C2 one figure, redesigned (fig 2's arms folded into fig 1's panels) | 30 | 8 pp |
| C1 one figure, stacked (shared caption only) | 53 | 9 pp |
| C0 two figures, priced from their real aspect ratio | 57 | 9 pp |
| two figures at the assumed 38+38 budget | 76 | 9 pp |

**Consolidating the two figures into one buys 1 page: 9 pp → 8 pp.**

**C1 is worth nothing and should not be attempted.** Stacking the two figures under a shared
caption saves one caption and nothing else: 57 slots become 53, and the compile does not move.
The panel area is unchanged, because stacking removes no panel. Only C2 saves area, and it
does so for a reason about the data rather than the layout — figure 2's early and delayed
arms are the same numbers figure 1 already plots (`EARLY` and `DELAY` in `make_figures.py`
feed both), so folding its treated arm into figure 1's per-size panels duplicates nothing
and loses nothing.

## How much slack C2 fits by

Bisecting a single spanning float: the paper is 8 pp up to **46 slots** and 9 pp above it.
C2 is estimated at **30 slots**, so it clears the turnover by **16 slots (0.15 page)**.

That margin is the number to design against: 16 slots is 106 pt of float height. Inverting the aspect ratio, a merged figure
drawn 7.1 in wide may be at most **4.45 in** tall before the page comes back
(C2 assumes 2.80 in).
**The consolidation must therefore be drawn and re-compiled, not assumed.** The slot figure
for C2 is derived from an aspect ratio chosen on paper; every other number in this file is
measured.

## 4. What the unanswered CFP question is now worth

`notes/interpscience-format-query.md` asks the workshop whether appendices are permitted.
That question was correctly judged non-load-bearing at a five-page overflow — it could not
change a plan that was failing by that much. **It is load-bearing now**, because the plan
fits with nothing to spare, and `move_cost` rows C–E move §2 material *into an appendix*.

Compiled with C2's figure and the §2 appendix move reversed — everything back in the body,
which is what a "no appendices" answer would force: **9 pp**, against 8 with it.

So the CFP answer decides whether the plan fits. Worth knowing before the figure is drawn, not after.

## 5. The formats stop being length-neutral at the last page

interpscience submits in **NeurIPS format against 9 pages**, so its fate has to be compiled
rather than carried over from the ACL number — and it differs. The same plan with C2 is
**9 pp (**fits**)**, but with no figures at all it is **9 pp**, so figures *are* a lever there;
C2 is simply **too big for it**. Bisecting: NeurIPS turns over at **36 slots** against ACL's 46.

| format | limit | turnover | C2 at 30 slots |
|:--|--:|--:|--:|
| ACL / BabyLM | 8 | 46 slots | 8 pp (**fits**) |
| NeurIPS / interpscience | 9 | 36 slots | 9 pp (**fits**) |

**Length-neutrality was a real result and it does not extend this far.** It held while both
formats failed by the same five pages; at the margin they diverge, because NeurIPS is single
column and a float there displaces text it cannot pack around. The practical consequence is a
tighter figure: a merged figure must come in under **36 slots** for interpscience versus **46**
for BabyLM — the same design decision, with the stricter of the two constraints binding if both
venues are still live. In inches at 7.1 in wide, that is **3.42 in** tall rather than 4.45.

## What this means for §3 and §4

**For ACL/BabyLM the figure consolidation is the difference between 9 pages and the 8-page limit.**
It is not a marginal saving to be banked alongside prose edits — on that route it is the move
that closes the gap, and everything the estimate treated as a remaining ~0.9-page residue to
be found in §3 and §4 is absorbed by drawing one figure instead of two.

**So on the BabyLM route, §3 and §4 prose should not be touched at all yet.** Draw the merged
figure, re-run this file against the real float, and only then ask whether any prose cut is
still needed. Cutting §3/§4 prose first would spend the argument's substance on a page the
layout can supply — and the two are not interchangeable, since prose cuts are irreversible in
review and a figure redesign is not.

**On the interpscience route the same conclusion holds, with a tighter figure** (§5): C2 as sized
is 9 pp there, but the format's turnover is 36 slots rather than 46, and below it the plan
is 9 pp against a 9-page limit. So that route also needs no §3/§4 prose cut — it needs the
merged figure drawn smaller. Neither venue's last page comes out of the argument.

Caveat, and it is the load-bearing one: C2's 30 slots is the only modelled number in the
chain — an aspect ratio chosen on paper. Against BabyLM's 46-slot turnover it has 16 slots
of room; against interpscience's 36 it is **-6 slots OVER**, which is why that route needs the figure
drawn to about 3.42 in rather than the 2.80 C2 assumes — **shorter than figure 1 alone is today
(2.45 in)**, which is the real design constraint and not an obvious one. Draw it, re-run
this file against the real float, and confirm before either target is treated as met.
