"""Remove em dashes from the typeset prose, rewriting each site so it stays grammatical.

An em dash does several different grammatical jobs, so a blanket substitution produces
comma splices wherever it was joining two independent clauses. The rules below are ordered
by how confidently the job can be identified from the following words, and **every site is
reported with before/after context** so the ones the rules get wrong can be hand-fixed
afterwards. That is the intended workflow: the script guarantees nothing but the dash sites
change, and a human settles the judgement calls.

Scope is the text that is actually typeset -- the eight section body ranges, the appendix,
and the abstract. Drafting apparatus in the same files (audit tables, length notes) is left
alone because it never reaches a PDF. The V2 replacement stubs live in `move_cost.py`,
`cut_dependencies.py` and `two_versions.py` and are edited by hand, being few.

Backups are timestamped `.bak` copies via the project's archive convention (this repo is
not under version control, so an unreviewable in-place rewrite would be unrecoverable).

  --apply   write the changes; without it, report only (default is a dry run)

Writes logs/em_dash_pass.md.
"""
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from page_estimate import SECTIONS, APPARATUS  # noqa: E402

OUT = "logs/em_dash_pass.md"

# (label, path, start regex, stop regex) for every range that is typeset.
RANGES = [(lab, p, s, e) for lab, p, s, e, _ in SECTIONS]
RANGES += [
    ("Appendix", "notes/draft-appendix.md", r"^# Appendix A", r"^## Notes"),
    ("Abstract", "notes/abstract-draft.md", r"^## Draft", r"^---\s*$"),
]

# Words that, following the dash, identify its job well enough to fix mechanically.
COORD = r"(and|but|so|or|yet|nor)"
RELATIVE = r"(which|who|whose|where|whom)"
SUBORD = r"(because|since|although|though|while|whereas|if|unless|after|before)"

RULES = [
    # A coordinating conjunction after the dash: the dash is doing a comma's job.
    (rf"\s*—\s+(?={COORD}\b)", ", ", "coordinating conjunction"),
    # A relative pronoun: non-restrictive clause, comma.
    (rf"\s*—\s+(?={RELATIVE}\b)", ", ", "relative clause"),
    # A subordinator: comma.
    (rf"\s*—\s+(?={SUBORD}\b)", ", ", "subordinate clause"),
    # A present participle ("— inverting our prediction"): comma. Requires lower case and
    # a following SPACE: `\w+ing\b` alone also matches the "Pruning" of a title like
    # "Pruning-based localization", turning a heading's dash into a comma.
    (r"\s*—\s+(?=[a-z]+ing\s)", ", ", "participial phrase"),
    # "— not X" / "— unlike X": contrastive tag, comma.
    (r"\s*—\s+(?=(not|never|no|unlike|rather|instead|except)\b)", ", ",
     "contrastive tag"),
]

# A heading's dash is a title separator, not punctuation. A colon is the usual
# replacement, but not when the title already contains one ("Appendix A - X: Y" would
# become "Appendix A: X: Y"), so a titling period is used instead.
HEADING = re.compile(r"^(#{1,6} .*?)\s*—\s+(.*)$", re.S)


# Subjects that signal what follows the dash is an independent clause, which needs a
# semicolon rather than a colon. A colon after an independent clause reads as an
# announcement and is wrong more often than not.
CLAUSE_START = r"(it|they|we|this|these|those|there|he|she|you|one)\b"


def paired(text):
    """Turn a balanced pair of dashes into parentheses.

    Operates on a whole PARAGRAPH, not a line: these drafts are hard-wrapped, so a
    parenthetical routinely opens on one line and closes on the next. Line-by-line
    matching sees two unrelated dashes, converts each separately, and silently produces
    an ungrammatical sentence -- which is exactly what the first version of this script
    did. Newlines inside the captured span are preserved, so the paragraph keeps its
    shape.
    """
    if text.count("—") < 2:
        return text, 0
    m = re.match(r"^(.*?)\s*—\s*(.+?)\s*—\s*(.*)$", text, re.S)
    if not m:
        return text, 0
    head, mid, tail = m.groups()
    if any(c in mid for c in "()[]") or not tail.strip():
        return text, 0
    # Two dashes in a paragraph are not necessarily a pair: they may be single dashes in
    # consecutive sentences, and bracketing across a sentence boundary would produce
    # "X (Y. Z) W". A parenthetical also does not run on for a whole paragraph.
    if re.search(r"[.!?]\s+[A-Z]", mid) or len(mid) > 220:
        return text, 0
    joiner = "," if re.match(rf"^\s*({COORD}|{SUBORD})\b", tail) else ""
    return f"{head} ({mid}){joiner} {tail}", 1


# A table cell containing nothing but a dash is a not-applicable placeholder, not
# punctuation. Rewriting it produces "|: |". Masked out before any rule runs.
CELL = re.compile(r"(?<=\|)(\s*)—(\s*)(?=\|)")
SENTINEL = "PLACEHOLDER"


def convert(text):
    """Return (new_text, [rule names]) for one paragraph."""
    hits = []
    n_cells = len(CELL.findall(text))
    text = CELL.sub(rf"\g<1>{SENTINEL}\g<2>", text)

    m = HEADING.match(text)
    if m:
        head, tail = m.groups()
        if re.match(r"^(unlike|rather|instead|except|not|but|and|so)\b", tail):
            joiner = ", "          # the dash continues the sentence, it does not title
        else:
            joiner = ". " if ":" in tail else ": "
        return f"{head}{joiner}{tail}", ["heading separator"]

    # Repeat: a paragraph may hold more than one parenthetical, and converting only the
    # first leaves the second to fall through to a colon, giving two colons in one
    # sentence. Bounded rather than while-True so a pathological input cannot spin.
    for _ in range(4):
        text, n = paired(text)
        if not n:
            break
        hits.append("paired -> parentheses")
    for pat, rep, name in RULES:
        text, k = re.subn(pat, rep, text, flags=re.S)
        if k:
            hits += [name] * k
    # Residual: the dash joins two spans no rule could classify. If what follows is an
    # independent clause it takes a semicolon; otherwise it expands or names what came
    # before, and takes a colon. Both are flagged for review either way.
    text, k = re.subn(rf"\s*—\s+(?={CLAUSE_START})", "; ", text, flags=re.S)
    if k:
        hits += ["UNCLASSIFIED -> semicolon"] * k
    text, k = re.subn(r"\s*—\s+", ": ", text, flags=re.S)
    if k:
        hits += ["UNCLASSIFIED -> colon"] * k

    text = text.replace(SENTINEL, "—")
    if n_cells:
        hits += [f"{n_cells} placeholder cell(s) left alone"]
    return text, hits


def ranges_in(path):
    """Line spans of `path` that are typeset, so apparatus is never rewritten."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    spans = []
    for lab, p, start, stop in RANGES:
        if p != path:
            continue
        i = next((k for k, ln in enumerate(lines) if re.match(start, ln)), None)
        if i is None:
            continue
        ends = [k for k, ln in enumerate(lines)
                if k > i and (re.match(stop, ln) or APPARATUS.match(ln))]
        spans.append((lab, i, min(ends) if ends else len(lines)))
    return lines, spans


def main(apply_it):
    report, total, reviewed = [], 0, 0
    for path in sorted({p for _, p, _, _ in RANGES}):
        lines, spans = ranges_in(path)
        changed = False
        edits = {}          # original start index -> (lines consumed, replacement lines)
        for lab, i, j in spans:
            # Group into paragraphs so a parenthetical that spans a line break is seen
            # whole. Headings and table rows are their own unit: a dash in a table cell
            # is a within-cell separator and must not bond with one in the next row.
            blocks, cur, start = [], [], i
            for k in range(i, j):
                s = lines[k].strip()
                solo = (not s) or s.startswith("|") or s.startswith("#")
                if solo:
                    if cur:
                        blocks.append((start, cur))
                        cur = []
                    blocks.append((k, [lines[k]]))
                else:
                    if not cur:
                        start = k
                    cur.append(lines[k])
            if cur:
                blocks.append((start, cur))

            for at, blk in blocks:
                before = "\n".join(blk)
                if "—" not in before:
                    continue
                after, hits = convert(before)
                if after == before:
                    continue
                changed = True
                total += len(hits)
                flag = any("UNCLASSIFIED" in h for h in hits)
                reviewed += flag
                report.append((lab, path, at + 1, before, after, hits, flag))
                # Record the edit against the ORIGINAL index; do not splice `lines` here.
                # Splicing shifts every later index, and the block list was computed
                # against the unspliced file -- which duplicated lines rather than
                # replacing them. The rewrite is applied in one pass below instead.
                edits[at] = (len(blk), after.split("\n"))
        if changed and apply_it:
            # Single forward pass over the original list: emit either the replacement or
            # the untouched line, never both. No index can go stale because none is reused.
            out_lines, k = [], 0
            while k < len(lines):
                if k in edits:
                    consumed, replacement = edits[k]
                    out_lines += replacement
                    k += consumed
                else:
                    out_lines.append(lines[k])
                    k += 1
            # Integrity gate. This pass changes PUNCTUATION only, so the sequence of word
            # tokens must be bit-identical before and after. A duplicated, dropped or
            # misplaced line changes that sequence and is caught here rather than being
            # discovered later in a PDF -- which is how the previous version's
            # index-shifting bug reached the drafts.
            was = re.findall(r"[A-Za-z0-9]+", "\n".join(lines))
            now = re.findall(r"[A-Za-z0-9]+", "\n".join(out_lines))
            if was != now:
                i = next((k for k, (x, y) in enumerate(zip(was, now)) if x != y),
                         min(len(was), len(now)))
                raise SystemExit(
                    f"{path}: word stream changed at token {i} -- refusing to write.\n"
                    f"  before: ...{' '.join(was[max(0, i - 8):i + 8])}...\n"
                    f"  after : ...{' '.join(now[max(0, i - 8):i + 8])}...")

            dest = f"{path}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            shutil.copy2(path, dest)          # archive, never delete
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")

    out = [
        "# Em-dash removal pass",
        "",
        f"Source: `scripts/drop_em_dashes.py`{'' if apply_it else ' (DRY RUN)'}. "
        f"Scope is the typeset ranges only",
        "(eight sections, appendix, abstract); drafting apparatus in the same files is "
        "untouched.",
        "",
        f"**{total} dashes rewritten across {len(report)} lines. "
        f"{reviewed} flagged for review.**",
        "",
        "A flagged site is one no rule could classify, so it took the fallback colon. "
        "Those are the",
        "ones worth reading; the rest follow a conjunction, relative pronoun, participle "
        "or contrastive",
        "tag, where the replacement is not a judgement call.",
        "",
    ]
    for lab, path, ln, before, after, hits, flag in report:
        out += [
            f"### {'REVIEW ' if flag else ''}{lab} — `{os.path.basename(path)}`:{ln}",
            f"*{', '.join(sorted(set(hits)))}*",
            "",
            "```",
            f"- {before.strip()}",
            f"+ {after.strip()}",
            "```",
            "",
        ]
    os.makedirs("logs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"{total} dashes across {len(report)} lines; {reviewed} flagged REVIEW")
    print(f"report: {OUT}" + ("" if apply_it else "   (dry run, nothing written)"))


if __name__ == "__main__":
    main("--apply" in sys.argv)
