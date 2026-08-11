"""Build V1 Full as the arXiv package: every section, references, appendices.

Distinct from every other build in this project in one respect that matters: **references
are typeset**. All the length work deliberately excluded them, because every venue
considered excludes them from its page limit -- but that meant no compile ever exercised
the bibliography, and the drafts' citations sat as plain bracketed text with no BibTeX
behind them. That gap was invisible in a page count by construction.

Citations are rewritten from the drafts' textual form into natbib keys at build time by
`build_paper.CITATIONS`, which fails loudly on any bracket group it does not recognise, so
a citation added to a draft cannot reach the PDF as literal brackets with no bibliography
entry. Keys resolve against `paper/references.bib`; provenance is `logs/bibliography.md`.

Writes paper/v1_arxiv.pdf and logs/arxiv_build.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_paper import CITATIONS, main as compile_paper  # noqa: E402
from make_figures import FIG1_SIZE, FIG2_SIZE  # noqa: E402
from plan_compile import float_slots, measured_textwidth  # noqa: E402

OUT = "logs/arxiv_build.md"
# A and B ONLY. Appendix C holds material the condensed build moves out of the body;
# V1 moves nothing and carries all of it inline, so including C here would print it twice.
APPENDIX = ("notes/draft-appendix.md", r"^# Appendix A", r"^# Appendix C")


def main():
    compile_paper(style="acl", quiet=True)          # refresh probe.log for the geometry
    tw = measured_textwidth()
    figs = {"§3 Detect": (float_slots(FIG1_SIZE, tw), True),
            "§4 Reopen": (float_slots(FIG2_SIZE, tw), True)}

    body_only = compile_paper(style="acl", quiet=True, figure_lines=figs)
    with_refs = compile_paper(style="acl", quiet=True, figure_lines=figs,
                              references=True)
    full = compile_paper(style="acl", quiet=True, figure_lines=figs, real_figures=True,
                         references=True, appendix=APPENDIX, stem="v1_arxiv")

    n_keys = sum(len(v.split(",")) for v in CITATIONS.values())

    out = [
        "# V1 Full — the arXiv build",
        "",
        "Source: `scripts/build_arxiv.py` → `paper/v1_arxiv.pdf`. Every section as "
        "drafted, Related",
        "Work standalone, nothing moved to an appendix, figures unconsolidated — plus "
        "the two things",
        "every previous compile left out: **a typeset bibliography and the "
        "appendices**.",
        "",
        "| build | pages |",
        "|:--|--:|",
        f"| body only, references excluded (comparable to every length number) "
        f"| {body_only} pp |",
        f"| + references | {with_refs} pp |",
        f"| **+ appendices = the arXiv package** | **{full} pp** |",
        "",
        f"References cost **{with_refs - body_only} page"
        f"{'' if with_refs - body_only == 1 else 's'}** and the appendices "
        f"**{full - with_refs}**. Neither counts against any venue limit",
        "considered, which is exactly why neither had ever been compiled.",
        "",
        "## What was verified",
        "",
        f"**{n_keys} distinct works**, every one checked against a primary or "
        f"authoritative secondary",
        "source rather than written from memory — see `logs/bibliography.md` for the "
        "per-entry record,",
        "including the two corrections and the four entries that are not peer reviewed.",
        "",
        "The build refuses to produce a PDF if any citation is unresolved: "
        "`build_paper.cite()` raises",
        "on a bracket group missing from `CITATIONS`, bibtex's log is scanned for "
        "`Warning--` and",
        "missing entries, and the LaTeX log is scanned for `Citation ... undefined`. "
        "All three passed;",
        "the bibliography contains 18 `\\bibitem`s, one per work.",
        "",
        "## Still outstanding before this can be posted",
        "",
        "1. **`\\author{}` is empty.** Name, affiliation and contact email are needed "
        "from Ashwin. This",
        "   is the only remaining blocker in the source itself.",
        "2. **The submitter must be the account holder.** arXiv posting is public and "
        "effectively",
        "   permanent, needs Ashwin's account, and a first cs.CL/cs.LG submission may "
        "require",
        "   endorsement.",
    ]
    os.makedirs("logs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(out))


if __name__ == "__main__":
    main()
