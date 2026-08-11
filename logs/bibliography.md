# Bibliography — what was verified, what changed, and what is still flagged

Built 2026-08-09 for `paper/references.bib`. Every entry was checked against a primary or
authoritative secondary source. Nothing here was written from memory and left unchecked;
where a field could not be verified it was **omitted rather than guessed**, and that is
recorded below.

The whole exercise existed because of a structural blind spot: **every compile in this
project excluded references** ("as the venue excludes them"), so no page count ever
exercised the bibliography, and the drafts' 18 citations sat as plain bracketed text with
`paper/custom.bib` still holding the stock ACL template placeholders (Aho:72, APA:83,
Gusfield:97). A gap that cannot show up in the measurement you are taking does not
announce itself.

## Count

**18 distinct works**, not the ~14 estimated. All 18 appear in exactly one section,
Related Work. `build_paper.CITATIONS` maps the drafts' 11 bracket groups onto them.

## Priority four — cited method vs. what the paper actually does

These were flagged as unverified when the §5-deletion plan proposed dropping them. They
are **still cited in the prose** — Related Work's "Circuit identity" paragraph — because
that removal only ever existed in the `cut_dependencies` edit that fires when §5 is
dropped, and §5 stays in both V1 and V2.

The sentence attributes to them: *"causal ablation and activation patching in the
circuit-analysis tradition [Elhage; Wang; Meng] and representational similarity
[Kornblith]"*.

| work | fields | does it support the attributed method? |
|:--|:--|:--|
| Elhage et al. 2021 | Transformer Circuits Thread, 22 Dec 2021, 25 authors | **NO — see below** |
| Wang et al. | ICLR 2023 (arXiv 2211.00593, 2022) | **YES** — 26 heads in 7 classes found "using a combination of interpretability approaches relying on causal interventions" |
| Meng et al. 2022 | NeurIPS 2022, arXiv 2202.05262 | **YES** — causal tracing corrupts the subject token then restores selected internal activations; that *is* activation patching |
| Kornblith et al. 2019 | ICML 2019, PMLR 97, arXiv 1905.00414 | **YES** — this is the CKA paper; "representational similarity" is exactly its contribution |

### ⚠ Elhage et al. 2021 is misattributed

Verified from the source page: its method is **mathematical/analytical decomposition** —
reverse-engineering transformers by examining weight matrices directly and analysing
computational paths (QK/OV circuits, induction heads). It does **not** perform causal
ablation or activation patching, and does not present itself as doing so.

It is a correct citation for *"the circuit-analysis tradition"* and a wrong one for *"causal
ablation and activation patching"*. As written, the sentence groups all three under both.
**Suggested fix — a clause reorder, not a deletion:** cite Elhage for the tradition and
Wang/Meng for the methods, e.g. "…causal ablation and activation patching
`\citep{wang-etal-2023,meng-etal-2022}` in the circuit-analysis tradition
`\citep{elhage-etal-2021}`…". Not applied: it changes prose, and prose is the author's.

Two further notes on that sentence. It reads "three of those four probes do not
discriminate", mapping four citations onto four probes — but the paper's four probes are
mean-ablation, whole-layer ablation, CKA and single-head patching, with pruning a fifth
that was pre-screened out. The citation list and the probe list are not the same four, and
a reader may try to align them. And **Elhage is not peer reviewed** (see below), which is
worth knowing when it is the lead citation of a methods claim.

## Corrections the verification produced

| # | what changed | detail |
|--:|:--|:--|
| 1 | **Wang et al. year moves 2022 → 2023** | draft cites the arXiv preprint; the archival version is ICLR 2023. The rendered in-text year changes accordingly. |
| 2 | **Achille et al. title differs between versions** | arXiv 1711.08856 is *"Critical Learning Periods in Deep **Neural** Networks"*; the ICLR 2019 paper is *"Critical Learning Periods in Deep Networks"*. Cited as the ICLR version, with the arXiv title in a `note`. |
| 3 | **Five citations had no year at all** | Hubel & Wiesel → 1970, Lenneberg → 1967, Johnson & Newport → 1989, Lyle et al. → 2023, Berariu et al. → 2021. |

## Not peer reviewed — four entries

Stated in the entries rather than dressed up as conference papers:

- **Elhage et al. 2021** — Transformer Circuits Thread, a web publication.
- **Berariu et al. 2021** — arXiv 2106.00042; no archival venue located.
- **Power et al. 2022** — arXiv 2201.02177; presented at the ICLR 2021 MATH-AI workshop,
  which is non-archival.
- **Lenneberg 1967** — a book, so peer review does not apply; listed here only so the
  count of non-conference sources is not surprising.

## Content claims checked, not just fields

The Related Work draft carried its own warning that the citations were "named from
background knowledge… above all *that the paper says what I have it saying*". The
substantive claims were therefore checked, not only the bibliographic fields:

- **Achille et al.** — draft says a deficit imposed early (**blurred images**) leaves a
  permanent gap despite unlimited clean data, and the window's timing is a property of
  training rather than architecture. **Confirmed**: cataract-like blurring on CIFAR-10
  causes permanent performance loss; the paper's own contrast case is vertical flipping,
  which does *not* affect low-level statistics and is fully recoverable; timing is
  attributed to learning dynamics via Fisher Information / "Information Plasticity".
- **Ash & Adams** — warm-started models generalize worse than freshly initialized ones on
  the same final data. **Confirmed** (this is the paper's headline; Shrink & Perturb is
  its remedy).
- **Dohare et al.** — plasticity loss in continual learning, recoverable by re-injecting
  plasticity through resets. **Confirmed** (continual backprop reinitializes low-utility
  units).
- **Warstadt et al. 2020 / 2023, Linzen et al. 2016, Marvin & Linzen 2018, Bengio et al.
  2009, Power et al. 2022** — all used only for what they are canonically known for
  (BLiMP minimal pairs, the BabyLM setting, agreement attraction, curriculum learning,
  grokking). Fields taken verbatim from the ACL Anthology where applicable.

## Fields deliberately omitted rather than guessed

- **Kornblith et al. 2019** and **Lyle et al. 2023** — PMLR page ranges not verified;
  entries carry volume + URL instead.
- **Ash & Adams 2020** and **Meng et al. 2022** — NeurIPS page ranges not verified;
  entries carry the proceedings URL.

None of these affects citation resolution; all four render correctly.

## Separate finding — techniques cited nowhere at all

The Methods sections name these with **no citation of any kind**. For a workshop paper
under a page limit that is a defensible omission; for an arXiv "complete record" it is
worth a decision. Not added, because adding references changes prose:

`RoPE` / rotary embeddings (5 mentions) · `AdamW` · `FlashAttention`(-2) · `GELU` ·
pre-norm · `Holm` correction (4 mentions)

CKA is the one such technique that *is* covered, by Kornblith et al.

## Guard rails now in the build

`scripts/build_paper.py` refuses to emit a PDF with a broken bibliography:

1. `cite()` raises on any bracket group absent from `CITATIONS` — so a citation added to a
   draft cannot silently reach the PDF as literal brackets with no entry behind it;
2. bibtex's `.blg` is scanned for `Warning--` and "I couldn't find" lines;
3. the LaTeX log is scanned for `Citation ... undefined`.

All three pass. The bibliography contains 18 `\bibitem`s, one per work.
