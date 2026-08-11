# NeurIPS-format compile — is the workshop route a real option?

Source: `scripts/build_paper.py --style neurips`. Same body text as the ACL compile
(`page_estimate.body()`), same abstract, figure placeholders sized to the same *page
fraction* rather than the same line count, since NeurIPS is single-column. Style file
is **neurips_2025.sty** — the 2026 styles are not published yet (`Styles.zip` 404s);
NeurIPS geometry has been stable for years, but this is a proxy and should be re-run
when the 2026 file appears.

## **14 pages** — against 9 (interpscience long paper) and 8 (XAI4Science).

- Geometry pdfTeX reports: `fsize=10 baselineskip=10.95pt textheight=650.43pt columnwidth=397.48499
pt`
- Single column, 9in x 5.5in text block, and `\parskip` 5.5pt between paragraphs
  (NeurIPS sets `\parindent` to zero and separates paragraphs by space instead).
- Overfull hboxes: 0.
- References excluded here. Whether the workshops exclude them is **not stated in
  either CFP** — NeurIPS main-conference convention does, but that is an assumption,
  not a quoted rule, and it is worth one email before relying on it.
