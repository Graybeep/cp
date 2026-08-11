"""Page count for the drafts, measured against the real ACL/BabyLM template geometry.

`length_audit.py` divided words by an assumed 800/page. This replaces the assumption
with the template: it reads the geometry out of `acl.sty` (quoted below), measures
every line with the actual body font at the actual size, and greedily wraps each
paragraph to the actual column width. What it reports is line counts, which is what
a page is made of.

CORRECTED 2026-08-09 against an actual compile. The geometry below is now MEASURED
from pdfTeX, not read off acl.sty's source, because reading the source got it wrong:

    \\def\\@normalsize{\\@setsize\\normalsize{11pt}\\xpt\\@xpt}   % acl.sty:288

looks like a declaration that the body is 10 pt on 11 pt, and this script took it as
one. It is not. `\\@normalsize` and `\\@setsize` are LaTeX 2.09 commands; LaTeX2e sets
the body font through `\\@setfontsize`, which acl.sty never calls (grep it). Line 288
is dead code inherited from the old style file, and the body font actually comes from
`\\documentclass[11pt]{article}` -> size11.clo. pdfTeX reports, at \\begin{document}:

    fsize=10.95  baselineskip=13.6pt  textheight=704.60pt  columnwidth=219.09pt

So the leading is 13.6 pt, not 11, and the type is 10.95 pt, not 10. Both errors ran
the same way -- fewer lines per column AND wider words -- and together they understated
the paper by about a third (9.5 pages estimated, 13 compiled).

  A4 21.0 x 29.7 cm, 2.5 cm margins -> text block 16.0 x 24.7 cm
  columns  219.09 pt (measured; the 218.3 pt derivation was very nearly right)
  lines    704.60 pt / 13.6 pt leading -> 51 per column
  page     102 lines

Known biases, both small and in opposite directions:
  - LaTeX justifies with hyphenation and inter-word shrink; greedy no-hyphen
    wrapping needs slightly MORE lines. Measured overshoot on ACL-like prose is
    ~2-4%, so this script over-estimates a little.
  - It charges nothing for widow/orphan handling, float placement gaps, or
    equation displays, which cost real space. That under-estimates.
Net: treat +/- 0.3 page as the resolution. With the geometry corrected this script
lands at 12.6 against the compile's 13, and the residual is the second bias above --
float placement, and table cells that wrap to more than the one line charged here.

**`scripts/build_paper.py` is now the authority; this script is the fast approximation.**
Use it to size an edit, not to certify a page count.

Writes logs/page_estimate.md.
"""
import os
import re

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

# All four measured from the compile (see the docstring), not derived from acl.sty.
COL_PT = 219.09
LEADING = 13.6
TEXTHEIGHT_PT = 704.60
FONT_PT = 10.95
LINES_PER_COL = int(TEXTHEIGHT_PT // LEADING)
LINES_PER_PAGE = 2 * LINES_PER_COL

# first page overhead: \titlebox is 11\baselineskip and spans both columns,
# and the abstract sits below it in column 1
TITLE_BLOCK_LINES = 11 * 2
ABSTRACT_LINES = 24

# vertical cost of headings, in lines (the heading line plus its space above/below)
H_LINES = {1: 3.2, 2: 2.6, 3: 2.3}
# a table costs its rows, a header row, two rules and a caption
TABLE_OVERHEAD_LINES = 5
FIGURE_LINES = {"§3 Detect": 38, "§4 Reopen": 38}   # ~0.30 page each

T2P = TextToPath()
FONT = FontProperties(fname="C:/Windows/Fonts/times.ttf", size=FONT_PT)
_cache = {}

N = "notes"
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
OUT = "logs/page_estimate.md"


def width(s):
    if s not in _cache:
        _cache[s] = T2P.get_text_width_height_descent(s, FONT, False)[0] if s else 0.0
    return _cache[s]


def strip_md(s):
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]*)\*", r"\1", s)
    s = re.sub(r"~~([^~]*)~~", r"\1", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    return s


def wrap_lines(text, avail=COL_PT):
    """Greedy line count for one paragraph at `avail` points of column width."""
    words = strip_md(text).split()
    if not words:
        return 0
    lines, cur = 1, width(words[0])
    space = width("x x") - 2 * width("x")
    for w in words[1:]:
        ww = width(w)
        if cur + space + ww <= avail:
            cur += space + ww
        else:
            lines += 1
            cur = ww
    return lines


# Drafting apparatus lives in the same files as the paper text and must never be
# charged to a page budget. Naming only each section's own stop heading was a bug:
# adding a new note ABOVE the named stop silently made it "body" and inflated the
# estimate. Any apparatus heading now ends a section, whichever comes first.
APPARATUS = re.compile(
    r"^#{2,3} +(Redundancy check|Budget note|Length|Resolved|Cut whole|Notes? for"
    r"|Notes$|Drafting notes|Superseded|§[\d–—-]+ (sentence|audit))", re.I)


def body(path, start, stop):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    i = next(k for k, ln in enumerate(lines) if re.match(start, ln))
    ends = [k for k, ln in enumerate(lines)
            if k > i and (re.match(stop, ln) or APPARATUS.match(ln))]
    return lines[i:(min(ends) if ends else len(lines))]


def measure(lines, label):
    """Line count for a section body: paragraphs wrapped, headings and tables charged."""
    total, para, tables, in_table = 0.0, [], 0, False
    n_head = n_tab_rows = 0

    def flush():
        nonlocal total, para
        if para:
            total += wrap_lines(" ".join(para))
            para = []

    for ln in lines:
        s = ln.strip()
        if not s:
            flush()
            in_table = False
            continue
        h = re.match(r"^(#{1,3}) ", s)
        if h:
            flush()
            total += H_LINES[len(h.group(1))]
            n_head += 1
            continue
        if s.startswith("|"):
            flush()
            if re.match(r"^\|[:\- |]+\|$", s):
                continue
            if not in_table:
                total += TABLE_OVERHEAD_LINES
                tables += 1
                in_table = True
            total += 1                      # a table row is one line
            n_tab_rows += 1
            continue
        in_table = False
        li = re.match(r"^(\d+\.|[-*])\s+", s)
        if li:
            flush()
            # list items are indented ~1.5em and hang; charge the narrower measure
            total += wrap_lines(s[li.end():], COL_PT - 1.5 * 10) + 0.3
            continue
        if s.startswith(">"):
            flush()
            total += wrap_lines(s.lstrip("> "), COL_PT - 2 * 10)
            continue
        para.append(s)
    flush()
    total += FIGURE_LINES.get(label, 0)
    return total, n_head, tables, n_tab_rows


def main():
    rows, tot = [], 0.0
    for label, path, start, stop, budget in SECTIONS:
        lines_ = body(path, start, stop)
        n, heads, tabs, tabrows = measure(lines_, label)
        pages = n / LINES_PER_PAGE
        rows.append((label, n, heads, tabs, tabrows, FIGURE_LINES.get(label, 0),
                     pages, budget, pages - budget))
        tot += n

    body_pages = tot / LINES_PER_PAGE
    overhead = (TITLE_BLOCK_LINES + ABSTRACT_LINES) / LINES_PER_PAGE
    total_pages = body_pages + overhead
    capacity_lines = 8 * LINES_PER_PAGE - TITLE_BLOCK_LINES - ABSTRACT_LINES

    out = [
        "# Page estimate — approximation, calibrated against the compile",
        "",
        "Source: `scripts/page_estimate.py`. **Geometry corrected 2026-08-09**: it is now "
        "measured",
        "from pdfTeX (`scripts/build_paper.py`), not read off acl.sty, whose "
        "`\\def\\@normalsize`",
        "line is LaTeX 2.09 dead code and misled the earlier version into 10 pt on 11 pt.",
        f"Every line is measured in Times New Roman {FONT_PT} pt and greedily wrapped to "
        f"the real",
        f"column width of {COL_PT:.1f} pt at {LEADING} pt leading.",
        "",
        "**The compile is the authority (13 pages); this is the fast approximation used to "
        "size",
        "an edit.** The earlier version of this file reported ~9.5 pages and was wrong by "
        "a third.",
        "",
        f"- **{LINES_PER_COL} lines per column, {LINES_PER_PAGE} per page** "
        "(heightrounded).",
        f"- Title block {TITLE_BLOCK_LINES} line-slots + abstract ~{ABSTRACT_LINES} "
        f"→ **{capacity_lines} lines of body capacity** in 8 pages.",
        "- Resolution is about ±0.3 page (see the docstring's two opposing biases).",
        "",
        "| section | lines | heads | tables | rows | fig lines | est. pages | budget | over |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        out.append(f"| {r[0]} | {r[1]:.0f} | {r[2]} | {r[3]} | {r[4]} | {r[5] or '—'} | "
                   f"{r[6]:.2f} | {r[7]:.2f} | {r[8]:+.2f} |")
    out += [
        f"| **body total** | **{tot:.0f}** | | | | | **{body_pages:.2f}** | **8.00** | |",
        "",
        f"**Body {tot:.0f} lines against {capacity_lines} available "
        f"= {tot - capacity_lines:+.0f} lines "
        f"({(tot - capacity_lines) / LINES_PER_PAGE:+.2f} page).**",
        f"With the title block and abstract, the paper typesets to "
        f"**~{total_pages:.1f} pages** against a limit of 8 "
        f"(compiled: **13**, `logs/compile_result.md`).",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
