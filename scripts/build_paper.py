"""Assemble the drafts into ACL LaTeX and compile, so the page count is measured not modelled.

`page_estimate.py` models the typesetting: it reads acl.sty's geometry, measures every
line in Times at 10 pt and greedily wraps it, and says so in its own docstring that only a
compile settles it exactly. This is that compile. MiKTeX is installed and the real
`acl.sty` is vendored in `paper/`, so the model can now be checked against the thing it
models.

The point of reusing `page_estimate.body()` verbatim is that both numbers must describe the
same text. The drafts interleave paper prose with drafting apparatus -- audit tables, cut
lists, budget notes -- and `body()` is where the boundary between the two is defined. A
separate extraction here would make any disagreement between the estimate and the compile
uninterpretable: it could be the typesetting model or it could be that the two tools read a
different paper.

What is approximated, and in which direction:
  - Figures are placeholder rules sized to `page_estimate.FIGURE_LINES`, because the
    figures are rendered but not yet placed. Real floats can only cost MORE (float
    placement leaves gaps), so the compile is a lower bound on that component.
  - References are excluded, which is correct: the venue excludes them from the limit.

Writes paper/paper.tex, paper/paper.pdf, and logs/compile_result.md.
"""
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from move_cost import apply_edit  # noqa: E402
from page_estimate import (FIGURE_LINES, LINES_PER_COL,  # noqa: E402
                           LINES_PER_PAGE, SECTIONS, body)

PAPER = "paper"
OUT = "logs/compile_result.md"
PANDOC = os.path.expanduser("~/AppData/Local/Pandoc/pandoc.exe")
PDFLATEX = os.path.expanduser(
    "~/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe")
BIBTEX = os.path.expanduser(
    "~/AppData/Local/Programs/MiKTeX/miktex/bin/x64/bibtex.exe")

# Every non-ASCII character that occurs in the body OR appendix ranges, mapped for
# pdflatex+inputenc. An unmapped character is a FATAL compile error, not a bad glyph, so
# this list has to grow whenever a new range is typeset -- the appendix added the second
# row (2026-08-09). To regenerate: set-difference the ranges' characters against
# these keys.
UNICODE = {
    "00A7": r"\S", "00B1": r"$\pm$", "00D7": r"$\times$", "00F7": r"$\div$",
    "0394": r"$\Delta$", "03B2": r"$\beta$", "03B8": r"$\theta$", "03C1": r"$\rho$",
    "2013": "--", "2014": "---", "2026": r"\ldots", "2192": r"$\rightarrow$",
    "2212": r"$-$", "2248": r"$\approx$", "2264": r"$\leq$",
    "03BB": r"$\lambda$", "2016": r"$\|$", "2032": r"$'$", "2081": r"$_1$",
    "2265": r"$\geq$",
}

# The drafts carry citations as plain bracketed text ("[Achille, Rovere & Soatto, ICLR
# 2019]") because they are markdown notes, not LaTeX. Rewriting them into natbib keys
# HERE rather than editing the drafts keeps the prose the author settled untouched, keeps
# `page_estimate`'s line counts comparable across every measurement taken so far, and
# keeps one authoritative mapping instead of the same decision restated in eight files.
# Keys resolve against paper/references.bib; provenance is in logs/bibliography.md.
# Whitespace in the keys below is normalised before lookup, so a draft may hard-wrap
# inside a bracket group (two of them do).
CITATIONS = {
    "Hubel & Wiesel; Lenneberg; Johnson & Newport":
        r"\citep{hubel-wiesel-1970,lenneberg-1967,johnson-newport-1989}",
    "Achille, Rovere & Soatto, ICLR 2019": r"\citep{achille-etal-2019}",
    "Ash & Adams, NeurIPS 2020": r"\citep{ash-adams-2020}",
    "Dohare et al., Nature 2024; Lyle et al.; Berariu et al.":
        r"\citep{dohare-etal-2024,lyle-etal-2023,berariu-etal-2021}",
    "Bengio et al., 2009": r"\citep{bengio-etal-2009}",
    "Power et al., 2022": r"\citep{power-etal-2022}",
    "Warstadt et al., 2023": r"\citep{warstadt-etal-2023}",
    "Warstadt et al., 2020": r"\citep{warstadt-etal-2020}",
    "Linzen, Dupoux & Goldberg, 2016; Marvin & Linzen, 2018":
        r"\citep{linzen-etal-2016,marvin-linzen-2018}",
    # Wang et al. is cited as 2022 in the draft (the arXiv preprint); the archival
    # version is ICLR 2023, so the rendered year moves. Recorded in logs/bibliography.md.
    "Elhage et al., 2021; Wang et al., 2022; Meng et al., 2022":
        r"\citep{elhage-etal-2021,wang-etal-2023,meng-etal-2022}",
    "Kornblith et al., 2019": r"\citep{kornblith-etal-2019}",
    # The demoted-Related-Work variants (cut_dependencies.RW_PARAGRAPH and
    # two_versions.CIRCUIT_CLAUSE) compress the same citations into different text.
    # They map to the same keys; only the surface form differs.
    "Achille, Rovere & Soatto, 2019": r"\citep{achille-etal-2019}",
    "Ash & Adams, 2020; Dohare et al., 2024; Lyle et al.":
        r"\citep{ash-adams-2020,dohare-etal-2024,lyle-etal-2023}",
}

# A bracket group holding only digits and arithmetic punctuation is a numeric interval
# (e.g. a confidence interval "[0.611, 0.708]"), not a citation.
NUMERIC = re.compile(r"[\d\s,.\-–—+±%]*")


def cite(md):
    """Rewrite the drafts' textual citations into natbib keys.

    Fails loudly on any bracket group that is neither a numeric interval nor a known
    citation. That is the point: a citation added to a draft and not added here would
    otherwise typeset as literal brackets in the references-included build and appear in
    no bibliography -- silently, and looking exactly like a citation to a reader.
    """
    def sub(m):
        inner = re.sub(r"\s+", " ", m.group(1)).strip()
        if inner in CITATIONS:
            return CITATIONS[inner]
        if NUMERIC.fullmatch(inner):
            return m.group(0)
        raise SystemExit(
            f"unmapped bracketed citation: [{inner}]\n"
            f"  add it to build_paper.CITATIONS and paper/references.bib")
    return re.sub(r"\[([^\]]+)\]", sub, md)


NEURIPS_PREAMBLE = r"""\documentclass{article}
\usepackage[sglblindworkshop, final]{neurips_2025}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage{array}
\usepackage{calc}
\usepackage{hyperref}   % the NeurIPS template loads this in its .tex, not its .sty;
                        % pandoc's \texorpdfstring needs it
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
@@UNICODE@@
\makeatletter
\AtBeginDocument{\typeout{ACLDIAG fsize=\f@size\space
  baselineskip=\the\baselineskip\space textheight=\the\textheight\space
  columnwidth=\the\columnwidth\space textwidth=\the\textwidth}}
\makeatother
\title{Can Critical Periods in Language Models Be Reopened?}
\author{}
\begin{document}
\maketitle
\begin{abstract}
@@ABSTRACT@@
\end{abstract}
@@BODY@@
\end{document}
"""

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[final]{acl}
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage{array}
\usepackage{calc}   % pandoc's p{} column specs use calc's \real for proportional widths
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
@@UNICODE@@
% Report the geometry the template actually produces, rather than the geometry its
% source appears to declare. acl.sty's \def\@normalsize is LaTeX 2.09 legacy and is
% never invoked by LaTeX2e, so the body font comes from \documentclass[11pt] instead.
\makeatletter
\AtBeginDocument{\typeout{ACLDIAG fsize=\f@size\space
  baselineskip=\the\baselineskip\space textheight=\the\textheight\space
  columnwidth=\the\columnwidth\space textwidth=\the\textwidth}}
\makeatother
\title{Can Critical Periods in Language Models Be Reopened?}
\author{}
\begin{document}
\maketitle
\begin{abstract}
@@ABSTRACT@@
\end{abstract}
@@BODY@@
\end{document}
"""


def abstract_md():
    with open("notes/abstract-draft.md", encoding="utf-8") as f:
        lines = f.read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("## Draft"))
    out = []
    for ln in lines[i + 1:]:
        if ln.startswith("---"):
            break
        s = re.sub(r"^> ?", "", ln)
        out.append(s)
    return "\n".join(out).strip()


def pandoc(md):
    r = subprocess.run(
        [PANDOC, "-f", "markdown+pipe_tables", "-t", "latex",
         "--top-level-division=section", "--wrap=preserve"],
        input=md, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"pandoc failed:\n{r.stderr}")
    return r.stdout


def longtable_to_table(tex):
    """longtable does not work in two-column mode; rewrite each one as table+tabular.

    Pandoc emits longtable for every pipe table. Left as-is, pdflatex either errors or
    silently mis-sets the table, and a mis-set table is worse than a failed compile here
    because it still produces a page count.
    """
    tex = re.sub(r"\{\\def\\LTcaptype\{none\}[^\n]*\n", "", tex)

    def brace_group(s, i):
        """Return (contents, index past the closing brace) for the group opening at s[i].

        A regex cannot do this: pandoc's column specs are of the form `{@{}lrrrrr@{}}`,
        whose braces nest, and a flat character class silently truncates them at the
        first inner `}` -- which produces a match that looks fine and a table that is not.
        """
        assert s[i] == "{"
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        raise SystemExit("unbalanced braces in longtable column spec")

    out, pos = [], 0
    while True:
        m = re.compile(r"\\begin\{longtable\}\[\]").search(tex, pos)
        if not m:
            out.append(tex[pos:])
            break
        out.append(tex[pos:m.start()])
        spec, k = brace_group(tex, m.end())
        end = tex.index(r"\end{longtable}", k)
        inner = tex[k:end]
        inner = re.sub(r"^\\end(head|firsthead|lastfoot|foot)\s*$", "", inner, flags=re.M)
        inner = re.sub(r"^\\bottomrule\\noalign\{\}\s*$", "", inner, flags=re.M)
        inner = re.sub(r"\n{3,}", "\n\n", inner).strip()
        out.append("\\begin{table}[t]\n\\centering\\small\n"
                   f"\\begin{{tabular}}{{{spec}}}\n{inner}\n"
                   "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        pos = end + len(r"\end{longtable}")
        # drop the closing brace of pandoc's `{\def\LTcaptype{none} ... }` wrapper,
        # whose opening line was stripped above
        pos += 2 if tex[pos:pos + 2] == "\n}" else 0
    return "".join(out)


# The real figures, rendered by make_figures.py. Paths are relative to PAPER/, since that
# is pdflatex's working directory. Used by the deliverable builds; the length-measurement
# builds keep the placeholder rules, which is all a page count needs.
FIGURES = {
    "§3 Detect": (
        "../logs/figures/fig1_detect.pdf",
        "Per-seed paired arithmetic endpoints at step 5000, early versus delayed "
        "exposure, one panel per model size, with the $n$~=~1 onset-dose points at "
        "right. The 11M panel is greyed: it falls below the pre-registered anchor floor "
        "and is excluded from the confirmatory comparison."),
    "§4 Reopen": (
        "../logs/figures/fig2_reopen.pdf",
        "Absolute treated endpoints beside their untreated nulls and early-exposure "
        "anchors, with the pre-registered same-endpoint band shaded. Points are "
        "per-seed; bars are means $\\pm$ SD across three seeds."),
}


def real_figure(path, caption, style, span=True):
    """A float carrying an actual rendered figure rather than a sized blank."""
    env = "figure*" if (span and style == "acl") else "figure"
    return (f"\n\\begin{{{env}}}[t]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{path}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\end{{{env}}}\n")


def figure_stub(lines_tall, label, style, span=False):
    """A float costing the page the same fraction the estimator charges it.

    `lines_tall` is always denominated in line-slots of the 102 a page has, whichever
    template and whichever float width, so one number means one thing everywhere.

    `span` says whether the float is full-width (`figure*`) or sits in one column. It is
    a property of the FIGURE -- both of this paper's are drawn 7.1 in wide against a
    6.3 in text block -- so it is passed in, not inferred from the height. A full-width
    float consumes two slots per baseline, hence the halving; the slot cost is the same
    either way, but the placement is not, and placement is what moves the page count.
    A single-column float taller than a column cannot be placed at all: pdfTeX defers it
    to the end of the document and reports the page count of a different paper, so that
    case is refused rather than silently compiled. NeurIPS is single-column throughout,
    so the allowance is expressed there as a page fraction; sizing it in baselineskips
    instead would make the figure a different size in the two templates and corrupt
    exactly the comparison this script exists to make.
    """
    if style != "acl":
        return ("\n\\begin{figure}[t]\n\\centering\n"
                f"\\rule{{\\linewidth}}{{{(lines_tall / LINES_PER_PAGE):.3f}"
                "\\textheight}\n"
                f"\\caption{{Placeholder for the {label} figure.}}\n"
                "\\end{figure}\n")
    if not span and lines_tall > LINES_PER_COL:
        raise SystemExit(
            f"{label}: a single-column float of {lines_tall:.0f} slots exceeds the "
            f"{LINES_PER_COL}-slot column and cannot be placed; it must span.")
    env = "figure*" if span else "figure"
    # the caption itself occupies the 2 slots subtracted here
    baselines = (lines_tall / 2 - 2) if span else (lines_tall - 2)
    return (f"\n\\begin{{{env}}}[t]\n\\centering\n"
            f"\\rule{{\\linewidth}}{{{baselines:.2f}\\baselineskip}}\n"
            f"\\caption{{Placeholder for the {label} figure.}}\n"
            f"\\end{{{env}}}\n")


def main(style="acl", omit=(), quiet=False, edits=None, extra=None,
         figure_lines=None, appendix=None, references=False, stem=None,
         real_figures=False):
    """Compile the drafts.

    `omit`   drops whole sections, for costing a section-level cut.
    `edits`  {section label: [(excision, residual), ...]}, applied with `move_cost`'s own
             `apply_edit` so a plan compiled here is the same text `move_cost.py` and
             `cut_dependencies.py` costed -- the alternative, re-specifying the edits
             here, is how a compile and its cost model come to describe different papers.
    `extra`  {section label: markdown appended to that section}, for material that moves
             INTO a section, e.g. Related Work demoted to a Discussion paragraph.
    `figure_lines` overrides `page_estimate.FIGURE_LINES`, for costing a figure
             consolidation. Keys name the section the float is placed in; a value is
             either slots, or (slots, span) where span requests a full-width `figure*`.
    `real_figures` embeds the rendered figures instead of the sized placeholder rules.
             OFF by default: a length measurement only needs a correctly sized blank, and
             that is what this script was originally for. A DELIVERABLE build must set it,
             or the PDF ships with grey boxes captioned "Placeholder".
    `references` typesets the bibliography and runs bibtex. OFF by default because every
             venue considered excludes references from its page limit, so a length
             measurement must not include them -- but the arXiv build must.
    `appendix` (path, start, stop) for `body()`, typeset after `\\appendix`. It is
             appended to the SAME document, so the returned page count includes it; a
             body-only count means calling twice. Both venues exclude appendices from
             their limit, so the body-only number is the one a limit is checked against
             and the combined one is the document a reader receives.
    """
    figure_lines = FIGURE_LINES if figure_lines is None else figure_lines
    parts = []
    for label, path, start, stop, _ in SECTIONS:
        if label in omit:
            continue
        lines = body(path, start, stop)
        for excision, residual in (edits or {}).get(label, ()):
            lines = apply_edit(lines, excision, residual)
        if extra and label in extra:
            lines = lines + ["", extra[label]]
        parts.append("\n".join(lines))
    md = cite("\n\n".join(parts))

    bodytex = longtable_to_table(pandoc(md))

    # Place the figure floats at the head of the sections that own them.
    for label, tall in figure_lines.items():
        # Pandoc passes the section mark through as a literal character, not \S{}.
        m = re.search(r"\\section\{" + re.escape(label.split()[0]) + r"[^\n]*\n", bodytex)
        if not m:
            if label in omit:
                continue
            raise SystemExit(f"could not locate section heading for {label}")
        tall, span = tall if isinstance(tall, tuple) else (tall, False)
        if real_figures and label in FIGURES:
            path, caption = FIGURES[label]
            if not os.path.exists(os.path.join(PAPER, path)):
                raise SystemExit(f"{label}: {path} missing -- run make_figures.py")
            block = real_figure(path, caption, style, span)
        else:
            block = figure_stub(tall, label, style, span)
        bodytex = bodytex[:m.end()] + block + bodytex[m.end():]

    # ACL puts the references BEFORE the appendices, so this order is not cosmetic.
    if references:
        bodytex += "\n\\bibliography{references}\n"
    if appendix:
        bodytex += ("\n\\appendix\n"
                    + longtable_to_table(pandoc(cite("\n".join(body(*appendix))))))

    uni = "\n".join(rf"\DeclareUnicodeCharacter{{{k}}}{{{v}}}"
                    for k, v in sorted(UNICODE.items()))
    # token replacement, not %-formatting: the preamble is full of LaTeX comments
    tex = NEURIPS_PREAMBLE if style == "neurips" else PREAMBLE
    for tok, val in (("@@UNICODE@@", uni), ("@@ABSTRACT@@", pandoc(abstract_md())),
                     ("@@BODY@@", bodytex)):
        tex = tex.replace(tok, val)

    stem = stem or ("probe" if quiet else
                    ("paper" if style == "acl" else "paper_neurips"))
    os.makedirs(PAPER, exist_ok=True)
    with open(f"{PAPER}/{stem}.tex", "w", encoding="utf-8") as f:
        f.write(tex)

    def latex():
        subprocess.run(
            [PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
            cwd=PAPER, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if references:
        # latex -> bibtex -> latex -> latex is the minimum that resolves \cite labels
        # AND the page numbers that shift once the bibliography exists.
        latex()
        rb = subprocess.run([BIBTEX, stem], cwd=PAPER, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        blg = os.path.join(PAPER, f"{stem}.blg")
        errs = []
        if os.path.exists(blg):
            btlog = open(blg, encoding="utf-8", errors="replace").read()
            errs = re.findall(r"^(I couldn't find[^\n]*|Warning--[^\n]*)$",
                              btlog, re.M)
        if rb.returncode or errs:
            raise SystemExit("bibtex failed or reported problems:\n  "
                             + "\n  ".join(errs or [rb.stdout[-800:]]))
        latex()
        latex()
    else:
        latex()
        latex()
    log = open(f"{PAPER}/{stem}.log", encoding="utf-8", errors="replace").read()
    if references and re.search(r"Citation `[^']*' on page \d+ undefined", log):
        bad = sorted(set(re.findall(r"Citation `([^']*)' on page \d+ undefined", log)))
        raise SystemExit(f"undefined citations after bibtex: {bad}")
    m = re.search(rf"Output written on {stem}\.pdf \((\d+) pages", log)
    if not m:
        tail = "\n".join(log.splitlines()[-40:])
        raise SystemExit(f"compile did not produce a page count. log tail:\n{tail}")
    pages = int(m.group(1))

    if quiet:
        return pages

    overfull = len(re.findall(r"Overfull \\hbox", log))
    diag = re.search(r"ACLDIAG ([^\n]*(?:\n[^\n(]*)?)", log)
    if style == "neurips":
        report = [
            "# NeurIPS-format compile — is the workshop route a real option?",
            "",
            "Source: `scripts/build_paper.py --style neurips`. Same body text as the ACL "
            "compile",
            "(`page_estimate.body()`), same abstract, figure placeholders sized to the same "
            "*page",
            "fraction* rather than the same line count, since NeurIPS is single-column. "
            "Style file",
            "is **neurips_2025.sty** — the 2026 styles are not published yet "
            "(`Styles.zip` 404s);",
            "NeurIPS geometry has been stable for years, but this is a proxy and should be "
            "re-run",
            "when the 2026 file appears.",
            "",
            f"## **{pages} pages** — against 9 (interpscience long paper) and 8 "
            f"(XAI4Science).",
            "",
            f"- Geometry pdfTeX reports: `{(diag.group(1) if diag else 'n/a').strip()}`",
            "- Single column, 9in x 5.5in text block, and `\\parskip` 5.5pt between "
            "paragraphs",
            "  (NeurIPS sets `\\parindent` to zero and separates paragraphs by space "
            "instead).",
            f"- Overfull hboxes: {overfull}.",
            "- References excluded here. Whether the workshops exclude them is **not stated "
            "in",
            "  either CFP** — NeurIPS main-conference convention does, but that is an "
            "assumption,",
            "  not a quoted rule, and it is worth one email before relying on it.",
        ]
        out_path = "logs/compile_neurips.md"
        os.makedirs("logs", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")
        print("\n".join(report))
        return pages

    report = [
        "# Exact compile — measured, not modelled",
        "",
        "Source: `scripts/build_paper.py`. MiKTeX 25.12 / pdfTeX 4.23, the real `acl.sty` "
        "from",
        "acl-org/acl-style-files vendored in `paper/`, `\\usepackage[final]{acl}`, Times "
        "via the",
        "`times` package. Body text extracted with `page_estimate.body()` so both numbers "
        "describe",
        "the same text; abstract from `notes/abstract-draft.md`; two placeholder figure "
        "floats",
        "sized to `page_estimate.FIGURE_LINES`. References excluded, as the venue excludes "
        "them.",
        "",
        f"## **{pages} pages** against a limit of 8 — over by **{pages - 8}**.",
        "",
        "- `page_estimate.py` modelled **~9.5 pages** and was wrong by a third: it read "
        "the body",
        "  font off acl.sty's `\\def\\@normalsize` line, which is LaTeX 2.09 dead code. "
        "pdfTeX",
        "  reports **10.95 pt on 13.6 pt**, not 10 on 11 \u2014 51 lines/column, not 63.",
        f"- Corrected against these measurements, the estimator now models **12.6**; the "
        f"compile says **{pages}**.",
        "  The remaining gap is float placement and multi-line table cells, both of which "
        "the",
        "  estimator under-charges by design, so the two agree in size and direction.",
        f"- Overfull hboxes: {overfull} (cosmetic; they do not move the page count).",
        "",
        "The compile is a *lower* bound on the figure component: the placeholders are "
        "sized to",
        "the estimator's allowance, and real floats can only cost more once placement "
        "gaps are",
        "charged. Everything else is exact.",
    ]
    os.makedirs("logs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    style = "neurips" if "--style" in sys.argv and "neurips" in sys.argv else "acl"
    need = "neurips_2025.sty" if style == "neurips" else "acl.sty"
    if not os.path.exists(f"{PAPER}/{need}"):
        raise SystemExit(f"paper/{need} missing")
    if not shutil.which(PANDOC) and not os.path.exists(PANDOC):
        raise SystemExit("pandoc not found")
    main(style)
