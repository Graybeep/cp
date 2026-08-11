"""SUPERSEDED 2026-08-07 by scripts/page_estimate.py -- use that one.

This divides words by an assumed 800/page. `page_estimate.py` instead reads the
geometry out of `acl.sty` and measures every line in the real body font at the real
column width, which is both more accurate and differently biased: this script
over-charged tables and under-charged headings, and put the drafts at ~10.3 pages
where the template-based measure puts them at ~9.5. Kept as the record of how the
first (wrong) number was produced, since the write-up notes cite it.

Length audit of the paper drafts against the page budget in paper-outline.md.

Counts only *paper body* text: each section runs from its heading to the first
drafting-apparatus heading (a claim audit, a cut list, a resolved note, a budget
note). Those exist for us, not for the reader, and must not be charged to a budget.

Words are converted at WORDS_PER_PAGE, which is an estimate and the weakest part
of this script -- so tables and figures are counted separately rather than folded
into it, since neither scales with word count. Read the word column as the hard
number and the page column as an indication.

Writes logs/length_audit.md.
"""
import os
import re

WORDS_PER_PAGE = 800          # ACL two-column body text; +/- 10% is realistic
TABLE_ROW_PAGES = 0.011       # a table row plus its share of header/rule overhead
FIGURE_PAGES = {"§3 Detect": 0.30, "§4 Reopen": 0.30}   # fig 1 and fig 2

N = "notes"
# (label, file, start regex, stop regex, budget from paper-outline.md)
SECTIONS = [
    ("§1 Introduction", f"{N}/draft-sections-1-2.md", r"^# §1 ", r"^# §2 ", 0.60),
    ("§2 Methods", f"{N}/draft-sections-1-2.md", r"^# §2 ", r"^### §1–§2 audit", 1.25),
    ("§3 Detect", f"{N}/draft-sections-3-4.md", r"^# §3 ", r"^### §3 sentence", 1.85),
    ("§4 Reopen", f"{N}/draft-sections-3-4.md", r"^# §4 ", r"^### §4 sentence", 1.75),
    ("§5 Circuit identity", f"{N}/draft-section-5.md", r"^# §5 ", r"^### §5 sentence", 1.00),
    ("Related Work", f"{N}/draft-related-work.md", r"^# Related Work", r"^## Notes for", 0.45),
    ("§6 Agreement", f"{N}/draft-section-6.md", r"^# §6 ", r"^### §6 sentence", 0.65),
    ("§7–§9", f"{N}/draft-sections-7-9.md", r"^# §7 ", r"^## Budget note", 0.45),
]
OUT = "logs/length_audit.md"


def body(path, start, stop):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    i = next(k for k, ln in enumerate(lines) if re.match(start, ln))
    try:
        j = next(k for k, ln in enumerate(lines) if k > i and re.match(stop, ln))
    except StopIteration:
        j = len(lines)
    return lines[i:j]


def main():
    rows, tot_w, tot_p, tot_b = [], 0, 0.0, 0.0
    for label, path, start, stop, budget in SECTIONS:
        lines = body(path, start, stop)
        # a markdown table row: starts with | and is not the |:--|--:| rule
        table = [ln for ln in lines
                 if ln.strip().startswith("|") and not re.match(r"^\|[:\- |]+\|$", ln.strip())]
        words = sum(len(ln.split()) for ln in lines if ln not in table)
        pages = words / WORDS_PER_PAGE + len(table) * TABLE_ROW_PAGES \
            + FIGURE_PAGES.get(label, 0.0)
        rows.append((label, words, len(table), FIGURE_PAGES.get(label, 0.0),
                     pages, budget, pages - budget))
        tot_w += words
        tot_p += pages
        tot_b += budget

    lines_out = [
        "# Length audit — drafts against the page budget",
        "",
        f"Source: `scripts/length_audit.py`. Body text only (drafting apparatus — claim",
        "audits, cut lists, resolved notes — excluded). Conversion "
        f"{WORDS_PER_PAGE} words/page,",
        f"{TABLE_ROW_PAGES} page per table row, figures charged explicitly. **The word",
        "column is measured; the page column is an estimate and should not be read to",
        "two decimals.**",
        "",
        "| section | words | table rows | figures | est. pages | budget | over |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for label, w, t, f_, p, b, d in rows:
        lines_out.append(f"| {label} | {w:,} | {t} | {f_ or '—'} | {p:.2f} | {b:.2f} | "
                         f"{d:+.2f} |")
    lines_out.append(f"| **total** | **{tot_w:,}** | | | **{tot_p:.2f}** | "
                     f"**{tot_b:.2f}** | **{tot_p - tot_b:+.2f}** |")
    lines_out += [
        "",
        f"At {WORDS_PER_PAGE} words/page the drafts stand at ~{tot_p:.1f} pages against an "
        f"8.00-page limit.",
        "Even at a generous 900 words/page the body alone is "
        f"~{tot_w / 900 + sum(r[2] for r in rows) * TABLE_ROW_PAGES + sum(FIGURE_PAGES.values()):.1f}"
        " pages.",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")
    print("\n".join(lines_out))


if __name__ == "__main__":
    main()
