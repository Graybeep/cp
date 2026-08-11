# Which whole sections would have to leave — compiled, not extrapolated

Source: `scripts/section_menu.py`. Each row is a real pdfTeX compile of the drafts
with those sections omitted. References excluded in both formats.

| what is kept | ACL / BabyLM (limit 8) | NeurIPS workshop (limit 9) |
|:--|--:|--:|
| everything (current drafts) | 13 pp (+5) | 14 pp (+5) |
| drop §6 Agreement | 12 pp (+4) | 13 pp (+4) |
| drop §5 Circuit identity | 11 pp (+3) | 12 pp (+3) |
| drop §5 + §6 | 10 pp (+2) | 11 pp (+2) |
| drop §5 + §6 + Related Work | 9 pp (+1) | 10 pp (+1) |
| §1–§4 only (floor, not a proposal) | 8 pp (**fits**) | 9 pp (**fits**) |

## Reading

- The full draft is **13 pages in ACL** against 8 and **14 in NeurIPS** against 9. The workshop route does **not** rescue the length:
  the more generous limit is more than cancelled by the single-column layout and NeurIPS's
  inter-paragraph `\parskip`. Both need roughly the same *fraction* of the paper removed.
- This is why the decision is now about sections rather than sentences. `logs/move_cost.md`
  measured the sentence-level ceiling: ~8.3 words per line, so the overflow is ~3,800 words.

**Caveat on the NeurIPS column.** The style file is `neurips_2025.sty`; the 2026 styles
were not published at the time of writing. And neither workshop CFP states whether
references count toward the limit — assumed excluded here, per main-conference convention.
Confirm before relying on the exact margin; neither uncertainty is large enough to change
the conclusion, which is driven by a 5-page gap.
