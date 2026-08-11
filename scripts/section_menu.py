"""What each whole-section removal actually buys, compiled in both candidate formats.

`move_cost.py` settled that sentence-level edits cannot close a 4.5-page overflow. This
answers the question that replaces it: which *sections* would have to go, and does either
venue's limit become reachable. Every row is a real pdfTeX compile of the drafts with
those sections omitted -- not the per-section page estimates added up, which would double
count nothing but would still inherit the estimator's known under-charging of floats.

Both formats are costed because the two are not interchangeable and the difference is not
in the direction one would guess: NeurIPS is single-column with `\\parskip` between
paragraphs, and comes out LONGER than ACL's two columns despite a more generous limit.

Writes logs/section_menu.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_paper import main as compile_paper  # noqa: E402

OUT = "logs/section_menu.md"

# Limit per format. ACL/BabyLM is 8; the NeurIPS number is interpscience's long-paper 9,
# the most generous of the three workshops found (XAI4Science is 8, Interpretability for
# Discovery is 5 and is out of reach on any version of this paper).
LIMITS = {"acl": 8, "neurips": 9}

SCENARIOS = [
    ("everything (current drafts)", ()),
    ("drop §6 Agreement", ("§6 Agreement",)),
    ("drop §5 Circuit identity", ("§5 Circuit identity",)),
    ("drop §5 + §6", ("§5 Circuit identity", "§6 Agreement")),
    ("drop §5 + §6 + Related Work",
     ("§5 Circuit identity", "§6 Agreement", "Related Work")),
    # §1–§4 only. Not a proposal — a paper without a discussion or limitations section
    # is not submittable. It is here as the floor: what remains if every section the
    # length debate has treated as optional is gone.
    ("§1–§4 only (floor, not a proposal)",
     ("§5 Circuit identity", "§6 Agreement", "Related Work", "§7–§9")),
]


def main():
    rows = []
    for name, omit in SCENARIOS:
        got = {}
        for style in ("acl", "neurips"):
            got[style] = compile_paper(style=style, omit=omit, quiet=True)
        rows.append((name, omit, got))

    out = [
        "# Which whole sections would have to leave — compiled, not extrapolated",
        "",
        "Source: `scripts/section_menu.py`. Each row is a real pdfTeX compile of the "
        "drafts",
        "with those sections omitted. References excluded in both formats.",
        "",
        "| what is kept | ACL / BabyLM (limit 8) | NeurIPS workshop (limit 9) |",
        "|:--|--:|--:|",
    ]
    for name, omit, got in rows:
        cells = []
        for style in ("acl", "neurips"):
            p, lim = got[style], LIMITS[style]
            mark = "**fits**" if p <= lim else f"+{p - lim}"
            cells.append(f"{p} pp ({mark})")
        out.append(f"| {name} | {cells[0]} | {cells[1]} |")

    full = rows[0][2]
    out += [
        "",
        "## Reading",
        "",
        f"- The full draft is **{full['acl']} pages in ACL** against 8 and "
        f"**{full['neurips']} in NeurIPS** against 9. The workshop route does **not** "
        "rescue the length:",
        "  the more generous limit is more than cancelled by the single-column layout and "
        "NeurIPS's",
        "  inter-paragraph `\\parskip`. Both need roughly the same *fraction* of the paper "
        "removed.",
        "- This is why the decision is now about sections rather than sentences. "
        "`logs/move_cost.md`",
        "  measured the sentence-level ceiling: ~8.3 words per line, so the overflow is "
        "~3,800 words.",
        "",
        "**Caveat on the NeurIPS column.** The style file is `neurips_2025.sty`; the 2026 "
        "styles",
        "were not published at the time of writing. And neither workshop CFP states "
        "whether",
        "references count toward the limit — assumed excluded here, per main-conference "
        "convention.",
        "Confirm before relying on the exact margin; neither uncertainty is large enough "
        "to change",
        "the conclusion, which is driven by a 5-page gap.",
    ]
    os.makedirs("logs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
