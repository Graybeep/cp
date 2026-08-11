# V1 Full — the arXiv build

Source: `scripts/build_arxiv.py` → `paper/v1_arxiv.pdf`. Every section as drafted, Related
Work standalone, nothing moved to an appendix, figures unconsolidated — plus the two things
every previous compile left out: **a typeset bibliography and the appendices**.

| build | pages |
|:--|--:|
| body only, references excluded (comparable to every length number) | 12 pp |
| + references | 13 pp |
| **+ appendices = the arXiv package** | **15 pp** |

References cost **1 page** and the appendices **2**. Neither counts against any venue limit
considered, which is exactly why neither had ever been compiled.

## What was verified

**22 distinct works**, every one checked against a primary or authoritative secondary
source rather than written from memory — see `logs/bibliography.md` for the per-entry record,
including the two corrections and the four entries that are not peer reviewed.

The build refuses to produce a PDF if any citation is unresolved: `build_paper.cite()` raises
on a bracket group missing from `CITATIONS`, bibtex's log is scanned for `Warning--` and
missing entries, and the LaTeX log is scanned for `Citation ... undefined`. All three passed;
the bibliography contains 18 `\bibitem`s, one per work.

## Still outstanding before this can be posted

1. **`\author{}` is empty.** Name, affiliation and contact email are needed from Ashwin. This
   is the only remaining blocker in the source itself.
2. **The submitter must be the account holder.** arXiv posting is public and effectively
   permanent, needs Ashwin's account, and a first cs.CL/cs.LG submission may require
   endorsement.
