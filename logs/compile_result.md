# Exact compile — measured, not modelled

Source: `scripts/build_paper.py`. MiKTeX 25.12 / pdfTeX 4.23, the real `acl.sty` from
acl-org/acl-style-files vendored in `paper/`, `\usepackage[final]{acl}`, Times via the
`times` package. Body text extracted with `page_estimate.body()` so both numbers describe
the same text; abstract from `notes/abstract-draft.md`; two placeholder figure floats
sized to `page_estimate.FIGURE_LINES`. References excluded, as the venue excludes them.

## **15 pages** against a limit of 8 — over by **7**.

- `page_estimate.py` modelled **~9.5 pages** and was wrong by a third: it read the body
  font off acl.sty's `\def\@normalsize` line, which is LaTeX 2.09 dead code. pdfTeX
  reports **10.95 pt on 13.6 pt**, not 10 on 11 — 51 lines/column, not 63.
- Corrected against these measurements, the estimator now models **12.6**; the compile says **15**.
  The remaining gap is float placement and multi-line table cells, both of which the
  estimator under-charges by design, so the two agree in size and direction.
- Overfull hboxes: 19 (cosmetic; they do not move the page count).

The compile is a *lower* bound on the figure component: the placeholders are sized to
the estimator's allowance, and real floats can only cost more once placement gaps are
charged. Everything else is exact.
