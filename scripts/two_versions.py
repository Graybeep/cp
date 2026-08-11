"""Two builds of the same paper, neither of which drops a finding. Compiled, not modelled.

This replaces the section-drop approach costed in `plan_compile.py` and `section_menu.py`.
Those asked which sections would have to leave; both answers required deleting §5 or §6,
i.e. deleting results. The question here is different and better posed: **what does the
paper cost when nothing is cut and only apparatus is moved?**

  V1 FULL       every section as drafted, Related Work standalone, nothing in an
                appendix, figures unconsolidated. The permanent record; arXiv has no
                limit, so this build has no target and is not optimised against one.

  V2 CONDENSED  same claims, same sections, same figures' content. Only apparatus moves,
                and only by edits already measured claims-neutral in `move_cost.py`:
                §2's engineering settings / embedding-share arithmetic / un-estimability
                derivations to the appendix, claim 21's power numbers and §8's seed-count
                derivations to the appendix, Related Work demoted to a Discussion
                paragraph, and the two figures merged where they plot the same numbers.

**No venue limit is applied and none is quoted as a target.** The point of the exercise is
to learn what shape the paper is; choosing the venue from the number is the next step, not
this one. Quoting a limit here would invite trimming to it.

Three things this file does NOT do, each because the drafts already do it:
  * moving the pruning pre-screen and the classifier provenance to an appendix -- done
    2026-08-07, `notes/draft-appendix.md`; §5's body already carries only the
    one-sentence versions, so there is nothing left to move;
  * compressing §5 and §6 -- no *measured* compression of either exists. 1.85 and 0.65 are
    budget targets from the rebalance, never edits anyone applied and re-wrapped. What is
    demonstrated is claim 21's move, which is included.
  * drawing the merged figure -- it is not drawn, so V2 is compiled BOTH ways and the
    figure question is reported as a bracket rather than a single number.

Writes logs/two_versions.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_paper import main as compile_paper  # noqa: E402
from cut_dependencies import RW_PARAGRAPH  # noqa: E402
from make_figures import FIG1_SIZE, FIG2_SIZE  # noqa: E402
from move_cost import MOVES  # noqa: E402
from plan_compile import float_slots, measured_textwidth  # noqa: E402

OUT = "logs/two_versions.md"
# A, B and C. C carries what the condensed build's §2/§6/§8 stubs point at, so it belongs
# to V2 only -- V1 keeps that material in the body (see build_arxiv.py).
APPENDIX = ("notes/draft-appendix.md", r"^# Appendix A", r"^## Notes")
APPENDIX_V1 = ("notes/draft-appendix.md", r"^# Appendix A", r"^# Appendix C")

# Related Work demoted while §5 STAYS. `RW_PARAGRAPH` was drafted for the plan that
# deleted §5, so it drops the circuit-analysis citations -- correct there, wrong here:
# those citations position a section this version keeps. One clause restores them.
CIRCUIT_CLAUSE = (
    " Our circuit diagnosis sits in the causal-ablation and activation-patching tradition"
    " [Elhage et al., 2021; Wang et al., 2022; Meng et al., 2022] alongside"
    " representational similarity [Kornblith et al., 2019]; §5.1's negative"
    " methodological result is that three of those four probes do not discriminate on"
    " our contrast."
)


def condensed_edits():
    """Every move_cost row, imported rather than re-typed so the two cannot drift.

    `cut_dependencies.DEPS` is deliberately NOT applied: those edits resolve references
    to a DELETED §5, and §5 stays in both versions here. Applying them would silently
    strip §2.4's eval-contamination correction and §8's label-movers from a paper that
    still contains the section they support.
    """
    edits = {}
    for m in MOVES:
        if m.get("bundle"):
            continue
        edits.setdefault(m["label"], []).extend(m["edits"])
    return edits


def main():
    # `measured_textwidth` reads the diagnostic out of paper/probe.log, so a compile has
    # to have happened in THIS run -- reading a stale log would silently size the figures
    # against whatever geometry was last compiled.
    compile_paper(style="acl", quiet=True)
    tw = measured_textwidth()
    f1, f2 = float_slots(FIG1_SIZE, tw), float_slots(FIG2_SIZE, tw)
    c0 = {"§3 Detect": (f1, True), "§4 Reopen": (f2, True)}
    # Merged: fig 1's layout with fig 2's treated arm folded into its per-size panels.
    c2_slots = float_slots((FIG1_SIZE[0], 2.80), tw)
    c2 = {"§3 Detect": (c2_slots, True)}

    edits = condensed_edits()
    extra = {"§7–§9": RW_PARAGRAPH + CIRCUIT_CLAUSE}

    def build(appendix=APPENDIX, **kw):
        kw.setdefault("style", "acl")
        return (compile_paper(quiet=True, **kw),
                compile_paper(quiet=True, appendix=appendix, **kw))

    v1_body, v1_all = build(appendix=APPENDIX_V1, figure_lines=c0)
    v2_body, v2_all = build(omit=("Related Work",), edits=edits, extra=extra,
                            figure_lines=c2)
    v2c0_body, v2c0_all = build(omit=("Related Work",), edits=edits, extra=extra,
                                figure_lines=c0)
    # Keep a NAMED artifact for each version. Without this both were compiled to the
    # shared scratch stem and overwritten by the next build, so only the page counts
    # survived -- "two versions" existed as two measurements, not two PDFs.
    # The deliverable uses the two REAL figures, not the merged C2 placeholder: the merge
    # buys zero pages at this length (see the sweep above), and the merged figure has not
    # been drawn. Shipping a blank box to save nothing would be the wrong trade.
    v2_pdf = compile_paper(style="acl", omit=("Related Work",), edits=edits, extra=extra,
                           figure_lines=c0, real_figures=True, references=True,
                           appendix=APPENDIX, quiet=True, stem="v2_condensed")

    v1_np, _ = build(style="neurips",
                     figure_lines={k: v[0] for k, v in c0.items()})
    v2_np, _ = build(style="neurips", omit=("Related Work",), edits=edits, extra=extra,
                     figure_lines={"§3 Detect": c2_slots})

    out = [
        "# Two versions, no findings dropped — compiled",
        "",
        "Source: `scripts/two_versions.py`. Every number is a pdfTeX page count. **No "
        "venue limit is",
        "applied or targeted here** — the point is to find out what shape the paper is, "
        "and choosing a",
        "venue from the number is the next step rather than this one.",
        "",
        "## The two builds",
        "",
        "| | ACL two-column, body | + appendix | NeurIPS, body |",
        "|:--|--:|--:|--:|",
        f"| **V1 Full** — everything standalone, nothing moved | **{v1_body} pp** | "
        f"{v1_all} pp | {v1_np} pp |",
        f"| **V2 Condensed** — merged figure (modelled) | **{v2_body} pp** | "
        f"{v2_all} pp | {v2_np} pp |",
        f"| V2 Condensed — two figures, unmerged | {v2c0_body} pp | {v2c0_all} pp | — |",
        "",
        f"**V2 costs {v1_body - v2_body} page"
        f"{'' if v1_body - v2_body == 1 else 's'} less than V1 with the merged figure, "
        f"and {v1_body - v2c0_body} without it** — and no claim, section,",
        "table or figure differs between them. Only apparatus moved.",
        "",
        "## What V2 actually contains",
        "",
        "Identical to V1 in every result. The differences, all measured claims-neutral "
        "in",
        "`logs/move_cost.md` before being applied here:",
        "",
        "- **§2** — engineering settings, the embedding-share arithmetic and the three "
        "un-estimability",
        "  derivations move to the appendix. The cosine-decayed LR schedule and untied "
        "embeddings stay",
        "  in the body deliberately: claim 7 says closure *is* that schedule, and the "
        "reset intervention's",
        "  validity rests on the embedding/head separation.",
        "- **§6** — claim 21's power numbers move; the claim, its direction and its "
        "*suggestive* label stay,",
        "  because a labelled claim's qualifiers may not outrun it into supplementary "
        "material.",
        "- **§8** — the 14 / 59 / 82 seed-count derivations move; each limitation keeps "
        "its named experiment.",
        "- **Related Work** — demoted to a Discussion paragraph keeping the "
        "plasticity/warm-starting",
        "  positioning at near-full length. **The demoted paragraph needed a clause "
        "adding back**: the drafted",
        "  version (`cut_dependencies.RW_PARAGRAPH`) was written for the plan that "
        "*deleted* §5 and so drops",
        "  the circuit-analysis citations. §5 stays here, so they return "
        "(`CIRCUIT_CLAUSE`).",
        "- **§5 and §6 stay in the body, uncompressed.** See the caveat below — this is "
        "the one part of the",
        "  brief this file could not execute.",
        "",
        "## Appendix C — written, so the moves now land",
        "",
        "The §2, claim-21 and §8 moves do not delete their material, they **point at "
        "it**, and their",
        "stubs refer the reader to *Appendix C* five times. That appendix did not exist "
        "until now:",
        "`notes/draft-appendix.md` held only A and B, so the condensed build promised "
        "material it did",
        "not carry. **An appendix move that never lands is how content leaves a paper "
        "while every page",
        "count still looks correct** — the length machinery cannot detect it, because a "
        "dangling pointer",
        "costs no pages.",
        "",
        "Appendix C now carries all five: architecture and optimizer settings (C.1), "
        "parameter counts",
        "and the embedding share (C.2), the three un-estimability derivations (C.3), the "
        "attraction",
        "trajectory's power calculation (C.4), and the outstanding-test seed counts "
        "(C.5).",
        "",
        "**One stub was over-promising and was reworded rather than filled.** It said "
        "\"Appendix C gives",
        "the per-seed values\" for claim 21's trajectory — but the step-2500 evaluations "
        "were never",
        "retained per seed, only the paired aggregate, so those values exist nowhere in "
        "the logs. It now",
        "promises the power calculation, which does exist. The alternative was inventing "
        "six numbers.",
        "",
        "**Appendix C belongs to V2 only.** V1 moves nothing and carries that material "
        "inline, so it",
        "takes Appendices A and B; giving it C would print the same content twice.",
        "",
        "## Three things not done, and why",
        "",
        "1. **The pruning pre-screen and the classifier provenance were already moved** "
        "(2026-08-07,",
        "   `notes/draft-appendix.md` A and B). §5's body already carries only the "
        "one-sentence versions —",
        "   pruning named as the third non-discriminating probe with its reason, and the "
        "unreachable-label",
        "   fact. There is nothing left to move, and that move is already inside every "
        "number above.",
        "2. **No measured compression of §5 or §6 exists.** 1.85 and 0.65 are budget "
        "targets from the",
        "   2026-08-07 rebalance, not edits that were ever applied and re-wrapped; §5 "
        "currently measures",
        "   2.13 estimated pages against that 1.00 budget and §6 0.90 against 0.65. "
        "Compressing them is",
        "   real work with a real risk — both are dense with labelled claims whose "
        "qualifiers the label",
        "   rule forbids dropping — so it is not silently included in a number here. "
        "**If V2 needs to be",
        "   smaller, this is the next move and it has to be costed with `move_cost.py` "
        "first.**",
        "3. **The merged figure is not drawn** — and at this length it does not matter. "
        "Its slot cost is",
        f"   modelled ({c2_slots:.0f} slots at {FIG1_SIZE[0]} × 2.80 in), so V2 was "
        f"compiled both ways, and the two rows are",
        f"   **identical at {v2_body} pp**"
        + ("" if v2c0_body == v2_body else
           f" — {v2c0_body} unmerged versus {v2_body} merged") + ". Merging bought a "
        "page in the",
        "   section-drop plan only because that plan sat one line the wrong side of a "
        "page boundary;",
        "   V2 is nowhere near one. **So the C2 drawing is no longer on the critical "
        "path** — it should be",
        "   done for the reader (it removes real duplication) rather than for the page "
        "count.",
        "",
        "## Reading the number",
        "",
        f"**V2's body is {v2_body} pages, and the whole condensation bought "
        f"{v1_body - v2_body}.** That is the finding, and it is",
        "worth stating as a negative: every apparatus move available — three §2 moves, "
        "claim 21's power",
        "numbers, §8's derivations, a demoted Related Work, two appendices already "
        "banked, and a merged",
        f"figure — comes to {v1_body - v2_body} page between them. **The paper's length "
        f"is its results, not its packaging.**",
        "That is the honest reading of every length exercise in this project converging "
        "on the same",
        "answer from a fourth direction.",
        "",
        f"At {v2_body} pages it is not workshop-shaped on any of the three CFPs read "
        f"(limits 5, 8 and 9), and",
        "not by a margin any packaging decision reaches. The three options left are a "
        "venue whose limit",
        "fits this shape, an explicit length exception, or compressing §5/§6 — and only "
        "the last is still",
        "a *drafting* decision rather than a submission one. It is also **the only one "
        "that risks the claims**,",
        "which is why item 2 asks for it to be costed before it is attempted rather than "
        "after.",
        "",
        f"V1 is unaffected by all of it: {v1_body} pp of body, {v1_all} with the "
        f"appendices. Note that every",
        "number in this file still **excludes references**, for comparability with every "
        "other length",
        "measurement here. The actual arXiv package adds them — see "
        "`logs/arxiv_build.md`, which is",
        "the build that goes out.",
    ]

    os.makedirs("logs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(out))


if __name__ == "__main__":
    main()
