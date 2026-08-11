"""What a proposed cut or appendix-move actually buys, in lines, measured in context.

Two moves have now been costed by proxy and both under-delivered:

  1. The 2026-08-06 §3.2/§6 trims were sized with a words/800 rule that mis-attributed
     the overflow; §3 turned out to be +0.05, not +0.31, so the trim was made against a
     section that was never the problem.
  2. The 2026-08-07 appendix move of two §5 items bought ~0 page, because most of what
     moved was drafting apparatus the body never contained, and the body still owed a
     self-sufficient statement of each.

The lesson common to both is that a word count is not a line count and a *move* is not a
*cut*. This script therefore measures a move the only way that settles it: apply the edit
to the real draft text, re-wrap the affected section with `page_estimate`'s own measure,
and report the line delta -- separately for the naive version (delete the words) and the
honest version (delete the words, then pay back whatever stub the body still owes).

`residual` is the load-bearing field. An appendix relieves a narrative budget, not a
claims budget: if a claim stays in the ledger with a home in the body, the body must state
it self-sufficiently, because reviewers are not obliged to read an appendix. Set
`residual=None` only for a move where the body genuinely owes nothing back -- i.e. a cut.

Writes logs/move_cost.md.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from page_estimate import (COL_PT, FONT_PT, LEADING, LINES_PER_PAGE,  # noqa: E402
                           SECTIONS, body, measure)

OUT = "logs/move_cost.md"

# From logs/page_estimate.md, on the geometry corrected against the compile 2026-08-09:
# 1235 body lines against 770 of capacity in 8 pages. The pre-correction figures were
# 1153 against 962 (+191); they understated the problem by a factor of 2.4.
OVERFLOW = 465
CAPACITY = 770
BODY_WORDS = 8006

# Sections the moves touch, keyed by the label used in page_estimate.SECTIONS.
SECTION_BY_LABEL = {s[0]: s for s in SECTIONS}


def norm(s):
    """Collapse whitespace so a spec can be written without the draft's hard wraps."""
    return re.sub(r"\s+", " ", s).strip()


def apply_edit(lines, excision, residual):
    """Delete `excision` from the section body and splice `residual` in its place.

    Works on the joined text so a spec may span the draft's hard-wrapped source lines,
    then rebuilds paragraph blocks so `measure` re-wraps them from scratch -- which is
    the whole point: removing half a sentence does not remove half a line.
    """
    text = "\n".join(lines)
    flat = norm(text)
    target = norm(excision)
    if target not in flat:
        raise SystemExit(f"excision not found verbatim:\n  {target[:120]}...")

    # Rebuild block-by-block: normalise each paragraph, edit it, keep blank-line breaks.
    blocks, cur = [], []
    for ln in lines:
        if ln.strip():
            cur.append(ln)
        else:
            blocks.append(cur)
            cur = []
    blocks.append(cur)

    out, done = [], False
    for blk in blocks:
        if not blk:
            out.append([])
            continue
        joined = norm(" ".join(blk))
        if not done and target in joined:
            replaced = joined.replace(target, norm(residual or ""), 1)
            replaced = norm(replaced)
            out.append([replaced] if replaced else [])
            done = True
        else:
            out.append(blk)

    if not done:
        raise SystemExit("excision spans a paragraph break -- split it into two specs")

    rebuilt = []
    for blk in out:
        if blk:
            rebuilt.extend(blk)
        rebuilt.append("")
    return [ln for ln in rebuilt[:-1]]


def words(s):
    return len(norm(s).split()) if s else 0


def cost(label, edits):
    """Cost one scenario, which may bundle several edits to the same section."""
    _, path, start, stop, _ = SECTION_BY_LABEL[label]
    base_lines = body(path, start, stop)
    base_n = measure(base_lines, label)[0]

    naive, honest = base_lines, base_lines
    for excision, residual in edits:
        naive = apply_edit(naive, excision, None)
        honest = apply_edit(honest, excision, residual)
    return {
        "base": base_n,
        "naive": base_n - measure(naive, label)[0],
        "honest": base_n - measure(honest, label)[0],
        "w_out": sum(words(e) for e, _ in edits),
        "w_back": sum(words(r) for _, r in edits),
    }


MOVES = [
    dict(
        name="A. Claim 21 (§6 attraction-over-training) leaves the body",
        label="§6 Agreement",
        edits=[(
            """**Within a single run, attraction may also decline over training, but we do not
claim it.** Between steps 2500 and 5000 the isolated attraction cost falls +0.392 → +0.303
(paired, *n* = 6 at 23.2M, 5/6 seeds): marginal at *p* = 0.053, with 53 % power against
this effect size and an MDE of 0.125; roughly ten seeds would decide it. The mismatch cell
itself improves significantly over the same interval (+0.070, *p* = 0.041). We report the
direction and the power, and treat the trajectory as suggestive only.""",
            # A claim with a home in the body keeps its label's qualifiers in the body.
            # The §6 audit table requires the power numbers to travel with claim 21, so a
            # stub that drops them is not an appendix move, it is an unlabelled version.
            # "per-seed values" was the original stub and it promised data that does not
            # exist: the step-2500 evaluations were never retained per seed, only the
            # paired aggregate. A stub may not promise more than the appendix can carry.
            """**Attraction may also decline *within* a single run**, +0.392 → +0.303 between
steps 2500 and 5000 (paired, *n* = 6): marginal at *p* = 0.053 with 53 % power and an MDE
of 0.125, so we report it as suggestive only; Appendix C gives the power calculation.""",
        )],
    ),
    dict(
        name="B. §8 seed-count derivations (14 / 59 / 82) to the appendix",
        label="§7–§9",
        edits=[(
            """or 6 at a permissive threshold. This assumes delay eliminates the
high-acquisition mode; if it merely halves it, 59 are needed. The same test at 23M
requires 82 seeds and is out of reach.""",
            "(Appendix C derives the count and its threshold assumptions.)",
        )],
    ),
    # ---- §2 Methods. The hypothesis under test is that §2 is where an appendix move
    # finally works, because methods detail is not claims. Two constraints bound what may
    # leave, and both are consequences of results elsewhere in the paper rather than
    # matters of taste:
    #   * the cosine-decayed LR schedule STAYS -- claim 7 says closure at d50 is a
    #     property of that schedule, so a reader cannot check the mechanism argument with
    #     the schedule in supplementary material;
    #   * untied embeddings STAY -- the reset intervention's validity rests on the
    #     input representation and the prediction head being separable.
    dict(
        name="C. §2.1 engineering settings to the appendix (LR schedule + untying kept)",
        label="§2 Methods",
        edits=[(
            """Context length is 512 tokens. Positional information is rotary (θ = 10,000);
normalization is pre-norm; activations are GELU. **Input embeddings and the output
projection are untied**: a deliberate choice, since one of our interventions resets the
embedding layer and tied weights would confound an input-representation reset with a change
to the prediction head. Attention is `F.scaled_dot_product_attention`, which dispatches to
FlashAttention-2 kernels on our GPU. Training uses bf16 autocast, micro-batches of 8 × 512
tokens with 4 gradient accumulation steps, AdamW (β = 0.9/0.95, weight decay 0.1, gradient
clipping 1.0), a peak learning rate of 6e-4 with 300 warmup steps and cosine decay to
0.1 × peak. `torch.compile` is available in the trainer but was **disabled for the runs
reported here**. Every run reported here is 5,000 optimizer steps over 81.92M tokens.
Checkpoint spacing varies by run family (100 steps for the 23M arithmetic runs, 1,250 at
11M and 60M, 2,500 for the size sweep); it sets how finely a trajectory can be re-read
afterwards and nothing else, since delay and intervention triggers fire on exact step
count.""",
            """**Input embeddings and the output projection are untied**: a deliberate choice,
since one of our interventions resets the embedding layer and tied weights would confound
an input-representation reset with a change to the prediction head. Every run reported here
is 5,000 optimizer steps over 81.92M tokens at a peak learning rate of 6e-4 with 300 warmup
steps and **cosine decay to 0.1 × peak**, the schedule §4's mechanism result turns on.
Remaining architecture and optimizer settings, and per-family checkpoint spacing, are in
Appendix C.""",
        )],
    ),
    dict(
        name="D. §2.1 embedding-share arithmetic to the appendix (caveat kept)",
        label="§2 Methods",
        edits=[(
            """Both columns are given because untied embeddings make them tell different
stories. At vocabulary 16,384 the input embedding and output projection are 4.19M parameters
each at the smallest size (**72.7 % of that model, against 34.8 % at the largest**), so the
three configurations span 5.2× in total parameters but 12.5× in the parameters that do
computation. Size comparisons in this paper are stated in total parameters, which is the
axis our pre-registered analyses used; readers translating our scale results into
non-embedding terms should note that the effective range is wider than the labels suggest,
and that the smallest model is unusually embedding-dominated (§2.5).""",
            """Both columns are given because untied embeddings make them tell different
stories: the three configurations span 5.2× in total parameters but **12.5× in the
parameters that do computation**, and the smallest model is unusually embedding-dominated
(§2.5). Size comparisons here use total parameters, as pre-registered; Appendix C gives the
per-size breakdown.""",
        )],
    ),
    dict(
        name="E. §2.6 un-estimability derivations to the appendix",
        label="§2 Methods",
        edits=[(
            """After the floor exclusion the first has twelve observations with one per
seed × cell and no within-cell replication for a random effect. The second has a banned
denominator (recovery percentage divides by the seed-variable null), and two of its three
factors are constants. The third has both factors constant.""",
            """They fail for three different reasons (too few observations per cell, a banned
denominator, and constant factors), set out in Appendix C.""",
        )],
    ),
    dict(
        name="**C+D+E together: the whole §2 appendix move**",
        label="§2 Methods",
        bundle=True,   # restates C, D and E; must be excluded from the totals
        edits="ALL_SECTION_2",
    ),
]

# Expand the combined row from the individual §2 rows, so the bundle can never drift
# out of sync with its parts.
for _m in MOVES:
    if _m["edits"] == "ALL_SECTION_2":
        _m["edits"] = [e for other in MOVES
                       if other["label"] == "§2 Methods" and other is not _m
                       for e in other["edits"]]


def main():
    rows = []
    for m in MOVES:
        r = cost(m["label"], m["edits"])
        rows.append((m, r))

    out = [
        "# What each proposed move actually buys",
        "",
        "Source: `scripts/move_cost.py`, measured with `page_estimate.py`'s own wrap on "
        "the",
        f"geometry taken from the compile ({COL_PT:.1f} pt column, {FONT_PT} pt on "
        f"{LEADING} pt, {LINES_PER_PAGE} lines/page).",
        "Every figure below is a re-wrap of the edited draft, not a word count divided by "
        "anything.",
        "",
        "| move | section now | words out | words back | **cut** (delete outright) | "
        "**move** (delete, pay the stub back) |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for m, r in rows:
        out.append(
            f"| {m['name']} | {r['base']:.0f} ln | {r['w_out']} | {r['w_back']} | "
            f"−{r['naive']:.0f} ln ({r['naive'] / LINES_PER_PAGE:.3f} pg) | "
            f"−{r['honest']:.0f} ln ({r['honest'] / LINES_PER_PAGE:.3f} pg) |"
        )

    # Bundle rows restate other rows; summing them too would double-count.
    real = [(m, r) for m, r in rows if not m.get("bundle")]
    tot_naive = sum(r["naive"] for _, r in real)
    tot_honest = sum(r["honest"] for _, r in real)
    w_out = sum(r["w_out"] for _, r in real)
    rate = w_out / tot_naive          # words of prose per line, measured on these edits
    need = OVERFLOW * rate
    out += [
        f"| **all distinct moves (A–E)** | | {w_out} | "
        f"{sum(r['w_back'] for _, r in real)} | "
        f"**−{tot_naive:.0f} ln ({tot_naive / LINES_PER_PAGE:.2f} pg)** | "
        f"**−{tot_honest:.0f} ln ({tot_honest / LINES_PER_PAGE:.2f} pg)** |",
        "",
        f"Against the measured overflow of **+{OVERFLOW} lines "
        f"(+{OVERFLOW / LINES_PER_PAGE:.2f} page)**:",
        f"- as appendix *moves*, all of them together close "
        f"**{tot_honest / OVERFLOW * 100:.0f} %** of it;",
        f"- as outright *cuts*, **{tot_naive / OVERFLOW * 100:.0f} %**.",
        "",
        "## §2 is where the appendix finally works",
        "",
        "C+D+E buy more as an appendix **move** (23 lines) than claim 21 and the §8 "
        "derivations",
        "buy as outright **cuts** (15). That is the predicted asymmetry, and it is the "
        "first time",
        "the appendix has paid: §5 and §6 are dense with labelled claims, which must be "
        "restated",
        "in the body and so come back as a stub, whereas architecture constants, an "
        "embedding-share",
        "calculation and three un-estimability derivations are what supplementary material "
        "is for.",
        "",
        "Two items were held out of C deliberately, and both for reasons that come from "
        "results",
        "elsewhere rather than from taste: the **cosine-decayed LR schedule** (claim 7 says "
        "closure",
        "at d50 *is* that schedule, so a reader cannot check the mechanism argument with it "
        "in an",
        "appendix) and **untied embeddings** (the reset intervention's validity rests on "
        "the input",
        "representation and the prediction head being separable).",
        "",
        "## What size of move would work",
        "",
        f"These edits also calibrate the exchange rate, which is the number that should "
        f"size every later move: **{w_out} words of running prose bought {tot_naive:.0f} "
        f"lines = {rate:.1f} words per line.** (The whole-body average is "
        f"{BODY_WORDS / (OVERFLOW + CAPACITY):.1f}, but that "
        "counts headings, tables and figure slots, which hold lines without holding words. "
        "For deciding what a *paragraph* cut buys, the marginal rate above is the right "
        "one.)",
        "",
        f"So closing +{OVERFLOW} lines out of prose costs on the order of **{need:,.0f} "
        f"words** — against a body of **{BODY_WORDS:,} words**, i.e. about "
        f"**{need / BODY_WORDS * 100:.0f} % of everything written**. Every move costed here "
        f"removes {w_out} words between them, {w_out / need * 100:.0f} % of that. No "
        "collection of paragraph-level moves reaches it; the deficit is structural, and the "
        "remaining options are whole sections and the page limit itself "
        "(`logs/section_menu.md`).",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
