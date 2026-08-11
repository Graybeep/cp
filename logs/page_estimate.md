# Page estimate — approximation, calibrated against the compile

Source: `scripts/page_estimate.py`. **Geometry corrected 2026-08-09**: it is now measured
from pdfTeX (`scripts/build_paper.py`), not read off acl.sty, whose `\def\@normalsize`
line is LaTeX 2.09 dead code and misled the earlier version into 10 pt on 11 pt.
Every line is measured in Times New Roman 10.95 pt and greedily wrapped to the real
column width of 219.1 pt at 13.6 pt leading.

**The compile is the authority (13 pages); this is the fast approximation used to size
an edit.** The earlier version of this file reported ~9.5 pages and was wrong by a third.

- **51 lines per column, 102 per page** (heightrounded).
- Title block 22 line-slots + abstract ~24 → **770 lines of body capacity** in 8 pages.
- Resolution is about ±0.3 page (see the docstring's two opposing biases).

| section | lines | heads | tables | rows | fig lines | est. pages | budget | over |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| §1 Introduction | 72 | 1 | 0 | 0 | — | 0.71 | 0.60 | +0.11 |
| §2 Methods | 180 | 8 | 1 | 4 | — | 1.77 | 1.25 | +0.52 |
| §3 Detect | 251 | 4 | 2 | 16 | 38 | 2.47 | 1.85 | +0.62 |
| §4 Reopen | 198 | 5 | 2 | 9 | 38 | 1.94 | 1.75 | +0.19 |
| §5 Circuit identity | 217 | 5 | 1 | 7 | — | 2.13 | 1.00 | +1.13 |
| Related Work | 78 | 1 | 0 | 0 | — | 0.77 | 0.45 | +0.32 |
| §6 Agreement | 92 | 1 | 0 | 0 | — | 0.90 | 0.65 | +0.25 |
| §7–§9 | 142 | 6 | 0 | 0 | — | 1.40 | 0.45 | +0.95 |
| **body total** | **1231** | | | | | **12.07** | **8.00** | |

**Body 1231 lines against 770 available = +461 lines (+4.52 page).**
With the title block and abstract, the paper typesets to **~12.5 pages** against a limit of 8 (compiled: **13**, `logs/compile_result.md`).
