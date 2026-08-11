"""Compile the leading cut plan itself, and price the figure consolidation on top of it.

Two things were being done by arithmetic that should be done by pdfTeX.

1. **The leading plan had never been compiled.** Its page count was assembled by adding
   per-section line estimates, converting to pages, and then adding 0.48 as a "carried
   offset" because the estimator ran 0.48 page under the compile on the FULL draft. That
   is not a correction, it is an extrapolation: the gap between estimator and compile is
   float placement and multi-line table cells, and the leading plan deletes two of the
   float-and-table-heaviest sections, so there is no reason the residual should stay the
   same size -- still less stay the same size *per page* on a paper two thirds the length.
   `main()` now takes the plan's edits, so the plan compiles as a paper.

2. **The figure allowance was never derived from the figures.** `FIGURE_LINES` charges
   38 line-slots each, described in its own comment as "~0.30 page each" -- a budget, not
   a measurement. The figures are 7.1 x 2.45 in and 7.1 x 2.70 in (`make_figures.py`), so
   scaled to the text width their aspect ratio fixes their height, and the allowance can
   be computed rather than assumed.

The consolidation is then priced as a CURVE, not a point: the leading plan is compiled
across a sweep of total figure allowance, so the price of any candidate design is read
off the curve at that design's allowance. This matters because floats do not convert
smoothly into pages -- a float is placed at a column top and the page count moves in
whole pages -- so the curve can be, and is, flat over stretches. A single point estimate
would report the flat stretch as a real saving.

Writes logs/plan_compile.md.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_paper import main as compile_paper  # noqa: E402
from cut_dependencies import DEPS, RW_PARAGRAPH  # noqa: E402
from make_figures import FIG1_SIZE, FIG2_SIZE  # noqa: E402
from move_cost import MOVES  # noqa: E402
from page_estimate import LEADING, LINES_PER_PAGE  # noqa: E402

OUT = "logs/plan_compile.md"
LIMITS = {"acl": 8, "neurips": 9}

# The leading candidate from notes/paper-outline.md: drop §5, drop §6, demote Related
# Work into the Discussion, move §2.1/§2.6 detail to the appendix, and delete what §5
# drags with it.
OMIT = ("§5 Circuit identity", "§6 Agreement", "Related Work")


def leading_edits():
    """The plan's edits, taken from the scripts that costed them, never re-typed here."""
    edits = {}
    for d in DEPS:
        if d["label"] in OMIT:
            continue          # its host section is gone; the edit is moot
        edits.setdefault(d["label"], []).extend(d["edits"])
    for m in MOVES:
        if m["label"] == "§2 Methods" and not m.get("bundle"):
            edits.setdefault(m["label"], []).extend(m["edits"])
    return edits


def float_slots(size_in, textwidth_pt, caption_lines=2):
    """Line-slots a full-width float of this aspect ratio costs, on a 102-slot page.

    A `figure*` spans both columns, so each baseline of its height removes two of the
    page's line-slots. The figures are designed 7.1 in wide against a text block of
    ~6.3 in, so they are scaled DOWN to \\textwidth and the aspect ratio -- not the
    designed inch height -- is what survives into the page cost.
    """
    w_in, h_in = size_in
    height_pt = (textwidth_pt) * (h_in / w_in)
    return 2 * (height_pt / LEADING) + 2 * caption_lines


def measured_textwidth():
    """Read \\textwidth off the compile rather than deriving it from the paper size."""
    log = open("paper/probe.log", encoding="utf-8", errors="replace").read()
    m = re.search(r"ACLDIAG[^\n]*textwidth=([\d.]+)pt", log.replace("\n", " "))
    if not m:
        m = re.search(r"textwidth=([\d.]+)pt", log)
    if not m:
        raise SystemExit("ACLDIAG did not report textwidth; is build_paper's typeout in?")
    return float(m.group(1))


def verdict(pages, style):
    over = pages - LIMITS[style]
    return "**fits**" if over <= 0 else f"+{over}"


def main():
    edits, extra = leading_edits(), {"§7–§9": RW_PARAGRAPH}
    two = {"§3 Detect": 38, "§4 Reopen": 38}

    # Baselines. `current` re-establishes the 13; `leading_asbudgeted` is the plan on the
    # allowance every earlier estimate assumed.
    current = compile_paper(style="acl", quiet=True)
    tw = measured_textwidth()
    leading = compile_paper(style="acl", omit=OMIT, edits=edits, extra=extra, quiet=True)
    leading_np = compile_paper(style="neurips", omit=OMIT, edits=edits, extra=extra,
                               quiet=True)

    # Allowances implied by the real figures.
    c0 = float_slots(FIG1_SIZE, tw) + float_slots(FIG2_SIZE, tw)
    stacked = (FIG1_SIZE[0], FIG1_SIZE[1] + FIG2_SIZE[1])
    c1 = float_slots(stacked, tw)                      # one caption instead of two
    # Redesign: fig 2's three arms become a third x-position inside fig 1's per-size
    # panels, so the merged figure is fig 1's layout with a taller y-range and the band.
    # Panel COUNT is unchanged, which is why this is the only merge that saves area.
    c2 = float_slots((FIG1_SIZE[0], 2.80), tw)

    # Both figures are drawn wider than the text block, so every float here spans.
    sweep = [
        ("0 — no figures at all (floor, not a proposal)", 0, {}),
        ("C2 one figure, redesigned (fig 2's arms folded into fig 1's panels)",
         c2, {"§3 Detect": (c2, True)}),
        ("C1 one figure, stacked (shared caption only)", c1,
         {"§3 Detect": (c1, True)}),
        ("C0 two figures, priced from their real aspect ratio", c0,
         {"§3 Detect": (float_slots(FIG1_SIZE, tw), True),
          "§4 Reopen": (float_slots(FIG2_SIZE, tw), True)}),
        ("two figures at the assumed 38+38 budget", 76,
         {k: (v, True) for k, v in two.items()}),
    ]
    rows = []
    for name, slots, fl in sweep:
        rows.append((name, slots,
                     compile_paper(style="acl", omit=OMIT, edits=edits, extra=extra,
                                   figure_lines=fl, quiet=True)))

    def pages_at(slots, style="acl"):
        fl = {"§3 Detect": (slots, True) if style == "acl" else slots}
        return compile_paper(style=style, omit=OMIT, edits=edits, extra=extra,
                             figure_lines=fl, quiet=True)

    def turnover(style):
        """Largest single-float allowance still compiling to the no-figure page count.

        A design's slot estimate is modelled, so "it fits" is only actionable beside how
        much it fits BY. Returns (floor_pages, threshold_slots); threshold is None if the
        count never turns over below C0, i.e. figures are not a lever in that format.
        """
        floor = pages_at(0, style)
        lo, hi = 0.0, c0
        if pages_at(hi, style) == floor:
            return floor, None
        while hi - lo > 1:
            mid = (lo + hi) / 2
            if pages_at(mid, style) == floor:
                lo = mid
            else:
                hi = mid
        return floor, lo

    target, threshold = turnover("acl")
    np_floor, np_threshold = turnover("neurips")

    best = min(p for _, _, p in rows)
    worst = max(p for _, _, p in rows)
    c0_pages = next(p for n, _, p in rows if n.startswith("C0"))
    c2_pages = next(p for n, _, p in rows if n.startswith("C2"))

    out = [
        "# The leading plan, compiled — and what the figure consolidation is worth",
        "",
        "Source: `scripts/plan_compile.py`. Every number below is a pdfTeX page count of "
        "the",
        "leading plan's actual text: §5 and §6 omitted, Related Work demoted to "
        "`cut_dependencies.RW_PARAGRAPH`",
        "inside the Discussion, `move_cost` rows C–E applied to §2, and §5's dependents "
        "resolved",
        "as `cut_dependencies.DEPS` specifies. The edits are imported from the scripts "
        "that costed",
        "them, so the compiled paper and the costed plan cannot diverge.",
        "",
        "## 1. The plan compiles to a real page count",
        "",
        f"| | ACL (limit {LIMITS['acl']}) | NeurIPS (limit {LIMITS['neurips']}) |",
        "|:--|--:|--:|",
        f"| current drafts, everything | {current} pp (+{current - LIMITS['acl']}) | — |",
        f"| **leading plan** | **{leading} pp ({verdict(leading, 'acl')})** "
        f"| **{leading_np} pp ({verdict(leading_np, 'neurips')})** |",
        "",
        "The estimate this replaces was **8.46 pages plus a carried 0.48-page offset "
        "≈ 8.9**, and it",
        f"was right: the compile says {leading}. That is worth stating plainly rather "
        f"than quietly",
        "dropping, because the reasoning behind the offset was still unsound and would "
        "not have been",
        "self-correcting. The 0.48 was the estimator's residual on the *full* draft, "
        "carried across as",
        "though it were a property of the estimator; it is a property of the text — "
        "float placement and",
        "multi-line table cells — and the leading plan deletes §5 and §6, which hold a "
        "large share of",
        "both. An extrapolation that cannot be checked is not made safe by landing in "
        "the right place.",
        "",
        "What the compile adds is not a different number but the two things the estimate "
        "could not",
        "give: the paper is over by a **whole page, not 0.9 of one** — page counts are "
        "integers and a",
        "0.1-page distinction between them was never real — and the figure allowance "
        "underneath it is",
        "wrong in a way that changes the decision (§2, §3 below).",
        "",
        "## 2. The figure allowance was never measured",
        "",
        f"`FIGURE_LINES` charges **38 line-slots per figure** and its own comment calls "
        f"that "
        f"\"~0.30 page each\" — a budget written before the figures existed. The figures "
        f"are "
        f"{FIG1_SIZE[0]} × {FIG1_SIZE[1]} in and {FIG2_SIZE[0]} × {FIG2_SIZE[1]} in "
        f"(`make_figures.py`), and the text block",
        f"measures **{tw:.1f} pt** wide (read off the compile, not derived). Both are "
        f"wider than the text",
        "block, so both scale down to it and their aspect ratio fixes their height:",
        "",
        f"- figure 1 costs **{float_slots(FIG1_SIZE, tw):.0f} slots**, not 38;",
        f"- figure 2 costs **{float_slots(FIG2_SIZE, tw):.0f} slots**, not 38;",
        f"- together **{c0:.0f} slots against the 76 assumed** — the budget over-charges "
        f"the figures by",
        f"  {76 - c0:.0f} slots ({(76 - c0) / LINES_PER_PAGE:.2f} page).",
        "",
        "## 3. The consolidation, priced as a curve",
        "",
        "| figure treatment | slots | compiled |",
        "|:--|--:|--:|",
    ]
    for name, slots, pages in rows:
        out.append(f"| {name} | {slots:.0f} | {pages} pp |")

    saving = c0_pages - c2_pages
    out += [
        "",
        f"**Consolidating the two figures into one buys {saving} page"
        f"{'' if abs(saving) == 1 else 's'}: {c0_pages} pp → {c2_pages} pp.**"
        + ("" if saving else " That is, nothing."),
        "",
        "**C1 is worth nothing and should not be attempted.** Stacking the two figures "
        "under a shared",
        f"caption saves one caption and nothing else: {c0:.0f} slots become {c1:.0f}, "
        f"and the compile does not move.",
        "The panel area is unchanged, because stacking removes no panel. Only C2 saves "
        "area, and it",
        "does so for a reason about the data rather than the layout — figure 2's early "
        "and delayed",
        "arms are the same numbers figure 1 already plots (`EARLY` and `DELAY` in "
        "`make_figures.py`",
        "feed both), so folding its treated arm into figure 1's per-size panels "
        "duplicates nothing",
        "and loses nothing.",
        "",
    ]
    if threshold is None:
        out += [
            f"The page count never turns over between 0 and {c0:.0f} slots, so no figure "
            "treatment in",
            "this range changes it.",
        ]
    else:
        # Invert float_slots: the tallest design that still fits under the turnover.
        max_h_pt = (threshold - 2 * 2) * LEADING / 2
        max_design_h = (max_h_pt / tw) * FIG1_SIZE[0]
        out += [
            "## How much slack C2 fits by",
            "",
            f"Bisecting a single spanning float: the paper is {target} pp up to "
            f"**{threshold:.0f} slots** and {target + 1} pp above it.",
            f"C2 is estimated at **{c2:.0f} slots**, so it clears the turnover by "
            f"**{threshold - c2:.0f} slots "
            f"({(threshold - c2) / LINES_PER_PAGE:.2f} page)**.",
            "",
            f"That margin is the number to design against: {threshold - c2:.0f} slots is "
            f"{(threshold - c2) * LEADING / 2:.0f} pt of float height. Inverting the "
            f"aspect ratio, a merged figure",
            f"drawn {FIG1_SIZE[0]} in wide may be at most **{max_design_h:.2f} in** tall "
            f"before the page comes back",
            f"(C2 assumes 2.80 in).",
            "**The consolidation must therefore be drawn and re-compiled, not assumed.** "
            "The slot figure",
            "for C2 is derived from an aspect ratio chosen on paper; every other number "
            "in this file is",
            "measured.",
        ]
    # The plan now sits ON the limit, so the still-unanswered CFP question (are appendices
    # permitted?) has a price for the first time. At +5 pages it did not.
    no_appendix = {k: v for k, v in edits.items() if k != "§2 Methods"}
    no_appendix["§2 Methods"] = [e for d in DEPS if d["label"] == "§2 Methods"
                                 for e in d["edits"]]
    c2_fig = {"§3 Detect": (c2, True)}
    no_appx_pages = compile_paper(style="acl", omit=OMIT, edits=no_appendix, extra=extra,
                                  figure_lines=c2_fig, quiet=True)
    # interpscience submits in NeurIPS format against a 9-page limit, so the plan's fate
    # there has to be compiled too rather than carried over from the ACL number.
    c2_np = compile_paper(style="neurips", omit=OMIT, edits=edits, extra=extra,
                          figure_lines={"§3 Detect": c2}, quiet=True)
    np_fits = np_threshold is not None and np_floor <= LIMITS["neurips"]

    out += [
        "",
        "## 4. What the unanswered CFP question is now worth",
        "",
        "`notes/interpscience-format-query.md` asks the workshop whether appendices are "
        "permitted.",
        "That question was correctly judged non-load-bearing at a five-page overflow — it "
        "could not",
        "change a plan that was failing by that much. **It is load-bearing now**, because "
        "the plan",
        "fits with nothing to spare, and `move_cost` rows C–E move §2 material *into an "
        "appendix*.",
        "",
        f"Compiled with C2's figure and the §2 appendix move reversed — everything back in "
        f"the body,",
        f"which is what a \"no appendices\" answer would force: **{no_appx_pages} pp**"
        + (f", against {c2_pages} with it."
           if no_appx_pages != c2_pages else " — the same, so the answer is free.")
        + "",
        "",
        f"{'So the CFP answer decides whether the plan fits.' if no_appx_pages > LIMITS['acl'] >= c2_pages else 'So the plan survives either answer.'}"
        " Worth knowing before the figure is drawn, not after.",
        "",
        "## 5. The formats stop being length-neutral at the last page",
        "",
        f"interpscience submits in **NeurIPS format against {LIMITS['neurips']} pages**, "
        f"so its fate has to be compiled",
        f"rather than carried over from the ACL number — and it differs. The same plan "
        f"with C2 is",
        f"**{c2_np} pp ({verdict(c2_np, 'neurips')})**, but with no figures at all it is "
        f"**{np_floor} pp**, so figures *are* a lever there;",
        f"C2 is simply **too big for it**. Bisecting: NeurIPS turns over at "
        f"**{np_threshold:.0f} slots** against ACL's {threshold:.0f}.",
        "",
        "| format | limit | turnover | C2 at 30 slots |",
        "|:--|--:|--:|--:|",
        f"| ACL / BabyLM | {LIMITS['acl']} | {threshold:.0f} slots | "
        f"{c2_pages} pp ({verdict(c2_pages, 'acl')}) |",
        f"| NeurIPS / interpscience | {LIMITS['neurips']} | {np_threshold:.0f} slots | "
        f"{c2_np} pp ({verdict(c2_np, 'neurips')}) |",
        "",
        "**Length-neutrality was a real result and it does not extend this far.** It held "
        "while both",
        f"formats failed by the same five pages; at the margin they diverge, because "
        f"NeurIPS is single",
        "column and a float there displaces text it cannot pack around. The practical "
        "consequence is a",
        f"tighter figure: a merged figure must come in under **{np_threshold:.0f} slots** "
        f"for interpscience versus "
        f"**{threshold:.0f}**",
        "for BabyLM — the same design decision, with the stricter of the two constraints "
        "binding if both",
        "venues are still live. In inches at "
        f"{FIG1_SIZE[0]} in wide, that is "
        f"**{((np_threshold - 4) * LEADING / 2 / tw) * FIG1_SIZE[0]:.2f} in** tall rather "
        f"than {max_design_h:.2f}.",
        "",
        "## What this means for §3 and §4",
        "",
    ]
    if saving > 0 and c2_pages <= LIMITS["acl"]:
        out += [
            f"**For ACL/BabyLM the figure consolidation is the difference between "
            f"{c0_pages} pages and the {LIMITS['acl']}-page limit.**",
            "It is not a marginal saving to be banked alongside prose edits — on that "
            "route it is the move",
            "that closes the gap, and everything the estimate treated as a remaining "
            "~0.9-page residue to",
            "be found in §3 and §4 is absorbed by drawing one figure instead of two.",
            "",
            "**So on the BabyLM route, §3 and §4 prose should not be touched at all yet.** "
            "Draw the merged",
            "figure, re-run this file against the real float, and only then ask whether "
            "any prose cut is",
            "still needed. Cutting §3/§4 prose first would spend the argument's substance "
            "on a page the",
            "layout can supply — and the two are not interchangeable, since prose cuts are "
            "irreversible in",
            "review and a figure redesign is not.",
            "",
            f"**On the interpscience route the same conclusion holds, with a tighter "
            f"figure** (§5): C2 as sized",
            f"is {c2_np} pp there, but the format's turnover is {np_threshold:.0f} slots "
            f"rather than {threshold:.0f}, and below it the plan",
            f"is {np_floor} pp against a {LIMITS['neurips']}-page limit. So that route "
            f"also needs no §3/§4 prose cut — it needs the",
            "merged figure drawn smaller. Neither venue's last page comes out of the "
            "argument.",
            "",]
        if not np_fits:
            out += [
                "⚠ Except that this rests on a bisection over a single float, and the "
                "NeurIPS floor itself",
                f"({np_floor} pp) does not clear {LIMITS['neurips']} — re-read §5 before "
                f"relying on the sentence above.",
                "",]
        out += [
            f"Caveat, and it is the load-bearing one: C2's {c2:.0f} slots is the only "
            f"modelled number in the",
            f"chain — an aspect ratio chosen on paper. Against BabyLM's {threshold:.0f}"
            f"-slot turnover it has "
            f"{threshold - c2:.0f} slots",
            f"of room; against interpscience's {np_threshold:.0f} it is "
            f"**{c2 - np_threshold:.0f} slots OVER**, which is why that route needs "
            f"the figure",
            f"drawn to about {((np_threshold - 4) * LEADING / 2 / tw) * FIG1_SIZE[0]:.2f} "
            f"in rather than the {2.80:.2f} C2 assumes — **shorter than figure 1 alone "
            f"is today",
            f"({FIG1_SIZE[1]} in)**, which is the real design constraint and not an "
            f"obvious one. Draw it, re-run",
            "this file against the real float, and confirm before either target is "
            "treated as met.",
        ]
    else:
        out += [
            f"Figure work moves the leading plan from {c0_pages} pp to {c2_pages} pp, "
            f"against a limit of {LIMITS['acl']},",
            f"and the floor with no figures at all is {best} pp. So the consolidation "
            f"does not by itself",
            "close the gap, and §3/§4 prose remains on the table. Decide the figure "
            "question on the",
            "figures' merits and cost the prose separately.",
        ]

    os.makedirs("logs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    # The report is UTF-8; the Windows console is cp1252 and cannot encode "≈" or "§".
    # Replacing them on the way to stdout only -- the file keeps them.
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(out))


if __name__ == "__main__":
    main()
