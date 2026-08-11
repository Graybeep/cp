# Paper outline — section map and claim placement (v2, 2026-08-05)

Follows `week9-stats-backbone.md` (labels §3, families §4, consistency §5, owners §6,
banned phrasings §7, NOT-A-CLAIM §8, label-movers §9, prediction accountability §11) and
`abstract-draft.md` (anchors 1 → 11b → 4 → 7).

**Purpose.** Give every ledger row exactly one home, place every §11 obligation and every
§8 inert number at a named location, and record what is deliberately absent. The coverage
table at the bottom is the check; a row with no section is a bug in this document.

---

## Decisions — RESOLVED 2026-08-05 (v1 left these open)

**1. Venue and format: write to BabyLM's 8-page EMNLP-2026-workshop format; submission
target is a later cycle.** BabyLM 2026's direct deadline (July 15) and its ARR commitment
route (May 25) both closed before the write-up began — verified, not inferred. The content
target and the submission target therefore separate: **write to the format now, arXiv on
completion, submit to BabyLM 2027 when its CFP opens.** The format still fits the content;
only the calendar moved. A nearer-term ICLR/NeurIPS workshop with an open CFP is the
alternative path and has not been searched — it is a live option, not a rejected one.

**2. C5 (agreement attraction): compressed section — not appendix, not companion paper.**
It is the best-powered result in the project (72 models) with its own labelled claims;
an appendix is for supplementary support, and a companion paper splits a finding whose
methodological payoff is *stronger* read against the critical-period work. The
scoping-loss risk is what the ledger, the label rule and the banned-phrasings table were
built to absorb — declining to use them on the densest section would be inconsistent with
why they exist.

**3. The dissociation: a named subsection inside Detect (§3.3), not a standalone section.**
Merging it into the syntax prose recreates the belongs-to-no-cluster condition that hid
11b until the 2026-08-03 audit; a thin top-level section beside Detect/Reopen/Diagnose
reads as padding. A separately-headed subsection survives skimming, a ledger pass, and a
table-of-contents scan, which is the actual protection required.

> **Derived consequence — the syntax null moves under Detect too (§3.2).** 11b conjoins
> claims 1 and 11 and cannot be stated before both halves exist, so a §3.3 dissociation
> forces the LD null above it. This is coherent rather than a compromise: Detect is a
> *phase*, not a skill, and both results are detection-phase outcomes. It also removes one
> top-level heading, which the 8-page budget needs.

**Body order: Detect → Reopen → Diagnose → Agreement.** The abstract runs 1 → 11b → 4 → 7
(phenomenon → contrast → intervention → mechanism). The orders differ on purpose; do not
"fix" the mismatch by reordering either one.

### Page budget (8 pages, ACL two-column, references excluded)

| section | pages | notes |
|:--|--:|:--|
| §1 Introduction | 0.60 | trimmed 2026-08-06; it previewed results §3–§4 restate in full |
| §2 Methods | 1.25 | 2.3–2.6 compress first if over |
| §3 Detect (3.1 arith · 3.2 syntax · 3.3 dissociation) | 1.85 | trimmed 2026-08-06 via 3.2's probe list (one sentence per probe) |
| §4 Reopen | 1.75 | one table, one figure |
| §5 Circuit identity | 1.00 | |
| **Related Work** | **0.45** | **NEW 2026-08-06** — absorbs the §7.4 stub |
| §6 Agreement attraction | 0.65 | trimmed 2026-08-06; three-metrics argument compresses, claim 21 stays |
| §7 Discussion · §8 Limitations · §9 Conclusion | 0.45 | fits once §7.4 moves out; §7.2's three statements are not compressible |
| **total** | **8.00** | zero slack — first overflow comes out of §6, then §5 |
| *Appendices A–B* | *free* | *after the references, outside the limit — `notes/draft-appendix.md`* |

> **⚠ COMPILED 2026-08-09 — the paper is 13 pages against a limit of 8, and every section
> is over budget.** `scripts/build_paper.py` assembles the drafts into ACL LaTeX and runs
> pdfTeX (MiKTeX 25.12, the real `acl.sty` vendored in `paper/`); result in
> `logs/compile_result.md`. **This supersedes both earlier estimates** (~10.3, then ~9.5).
>
> **Why the estimates were wrong, and it is worth knowing.** `page_estimate.py` derived the
> geometry from acl.sty's source, where line 288 reads
> `\def\@normalsize{\@setsize\normalsize{11pt}\xpt\@xpt}` and looks like a declaration that
> the body is 10 pt on 11 pt. It is dead code: `\@normalsize` and `\@setsize` are LaTeX 2.09,
> LaTeX2e sets the body font through `\@setfontsize`, and acl.sty never calls it — the body
> font actually comes from `\documentclass[11pt]{article}`. pdfTeX reports **10.95 pt on
> 13.6 pt leading**, i.e. **51 lines/column, 102/page — not 63 and 126**. Both errors ran the
> same way (fewer lines per page, and wider type needing more of them), so the paper was
> understated by about a third. The estimator has been corrected against the compile, now
> lands at 12.6 against the compile's 13, and is demoted to a fast approximation for sizing
> an edit. **The compile is the authority.**
>
> **Net of the title block and abstract, 8 pages hold 770 body lines. The drafts are 1,235
> = +465 lines = +4.56 page.** Per-section overruns on the corrected geometry: §5 +1.13,
> §7–§9 +0.99, §3 +0.62, §2 +0.52, Related Work +0.32, §6 +0.25, §4 +0.19, §1 +0.11.
>
> **The "overflow is concentrated in two sections" reading is retired** — it was an artifact
> of the wrong geometry. §5 and §7–§9 are still the largest overruns, but they are 46 % of
> the problem rather than 79 %, no section is at budget, and no single section's removal
> fixes this.
>
> **Appendices are free, and have been used.** BabyLM 2026 caps papers at 8 pages and defers
> to EMNLP 2026 workshop standards, i.e. ACL's: appendices come after the references and are
> *"not required to fit within these page limits"*; references are unlimited. Two §5 items
> moved to `notes/draft-appendix.md` (pruning pre-screen, classifier provenance). **They
> bought ~0 page**, because the verbose parts were drafting apparatus the paper never
> contained, and the body must keep a self-sufficient statement of each — reviewers are not
> obliged to read an appendix. An appendix relieves a *narrative* budget, not a claims
> budget, and §5's overrun is claims.
>
> **Both candidate moves were costed 2026-08-09 and neither is worth executing.**
> `scripts/move_cost.py` → `logs/move_cost.md` applies each edit to the real draft and
> re-wraps it, rather than counting words. Claim 21 out of §6 buys **5 lines (0.05 page)**
> as an appendix move, 11 as an outright cut; the §8 seed-count derivations buy **3 lines
> (0.03 page)**, not the ~0.10 page the §7–§9 note assumed. Together: **2 %** of the
> overflow. Neither was executed.
>
> **The exchange rate is the number to plan with: ~8.3 words of prose per line** (measured
> on those two edits, not assumed). Closing 465 lines out of prose costs ~3,800 words
> against a body of ~8,000 — **about half of everything written**. §7–§9 was also checked
> against Related Work for redundancy (see `draft-sections-7-9.md`): only ~25 words overlap
> and both duplicate copies are Related Work's, so there is nothing to recover there either.
>
> **What remains is a scope decision, not a formatting one:** which results leave the paper.
> The §6 rule — cut whole rather than trim into an unlabelled version — is what should
> govern whatever goes.
>
> **⚠ THE VENUE ROUTE IS CLOSED AS A LENGTH FIX — compiled 2026-08-09, both formats.**
> `scripts/section_menu.py` → `logs/section_menu.md`. The drafts are **13 pages in ACL
> (limit 8)** and **14 in NeurIPS (limit 9, interpscience's long-paper track — the most
> generous of the three workshops found)**. The two formats are **length-neutral**: NeurIPS
> comes out exactly one page longer at every scenario tested, and its limit is exactly one
> page higher, so the overflow is **+5 pages either way**. NeurIPS's more generous limit is
> cancelled by its single column and its 5.5 pt `\parskip` between paragraphs. Choose the
> venue on fit and deadline; it will not buy a page.
>
> **What whole-section removals actually buy (compiled, not extrapolated):**
>
> | kept | ACL (8) | NeurIPS (9) |
> |:--|--:|--:|
> | everything | 13 (+5) | 14 (+5) |
> | drop §6 | 12 (+4) | 13 (+4) |
> | drop §5 | 11 (+3) | 12 (+3) |
> | drop §5 + §6 | 10 (+2) | 11 (+2) |
> | drop §5 + §6 + Related Work | 9 (+1) | 10 (+1) |
> | §1–§4 only | **8 fits** | **9 fits** |
>
> **Dropping both bonus clusters is not sufficient.** §5 + §6 together buy 3 pages of the 5.
> Adding Related Work buys a fourth and still leaves +1, and the only row that fits has no
> Discussion, Limitations or Conclusion — not a submittable paper. **So ~2 further pages
> must come from inside §1–§4 even after §5 and §6 go.**
>
> **§2 Methods was the one place an appendix move should work — TESTED 2026-08-09, and it
> does.** `logs/move_cost.md` rows C–E: §2.1's engineering settings, §2.1's embedding-share
> arithmetic and §2.6's un-estimability derivations together buy **23 lines (0.23 page) as
> an honest appendix move** and 46 as outright cuts — more as a *move* than claim 21 and the
> §8 derivations buy as *cuts*. First time the appendix has paid, and the reason is the
> predicted one: methods detail is not claims, so nothing has to come back as a stub.
> **Two items were deliberately held in the body**: the cosine-decayed LR schedule (claim 7
> says closure *is* that schedule) and untied embeddings (the reset intervention's validity
> rests on it).
>
> **§5's removal drags 0.38 page out with it — `section_menu.py` understated it**
> (`logs/cut_dependencies.md`). The menu removed only the section; §5 is also referred to
> from §1, §2.4, §3.1, §7.2, §8 and Related Work. 32 of those lines exist *only* to support
> §5 (§2.4's whole eval-contamination correction, two of §8's five costed label-movers,
> Related Work's circuit-identity citations) and delete with it; 7 more come from shortening
> the two passages that merely point at it. **The abstract needs no edit at all** — it
> carries no circuit content, which is the strongest indication §5 is genuinely severable.
> **And backbone §5.5 is satisfied, not violated**: the paired-null rule's underpowered half
> *is* §5's, so cutting §5 removes both halves. Recorded at §5.5 itself.
>
> **Related Work: DEMOTE, do not delete.** The standalone section is 78 lines; a Discussion
> paragraph keeping the plasticity/warm-starting distinction near full length, with Achille
> and grokking compressed to a clause each and the BabyLM/BLiMP/agreement citations pushed
> inline to §2.3 and §6, costs **27 lines**. So deletion buys 78 and demotion buys 51 — **the
> positioning function costs 0.26 page to keep, not 0.77.** Its own length note calls the
> plasticity paragraph the place "where a reader decides whether the schedule finding is a
> rediscovery"; deleting it saves a third of a page by leaving the most likely reviewer
> objection unanswered. Draft paragraph lives in `scripts/cut_dependencies.py:RW_PARAGRAPH`.
>
> **Revised arithmetic for the leading candidate** — drop §5, drop §6, demote Related Work,
> §2 appendix move, and delete §5's dependents. In lines: 1,231 − 217 (§5) − 92 (§6) − 51
> (RW demotion) − 23 (§2 move) − 31 (dependents outside the sections already removed) =
> **817 against a capacity of 770**.
>
> **⚠ SUPERSEDED 2026-08-09 (later): the section-drop approach below is RETIRED.** Every
> version of it required deleting §5 or §6 — i.e. deleting results. Replaced by a
> **two-version plan** in which nothing is cut from either build: **V1 Full** (everything
> standalone, arXiv, no limit) and **V2 Condensed** (same claims and sections, apparatus
> moved only). Compiled: `scripts/two_versions.py` → `logs/two_versions.md`.
>
> | | ACL body | + appendix | NeurIPS body |
> |:--|--:|--:|--:|
> | **V1 Full** | **13 pp** | 14 pp | 14 pp |
> | **V2 Condensed** | **12 pp** | 13 pp | 13 pp |
>
> **The entire condensation buys ONE page**, and the figure merge inside it buys **zero**
> (V2 is 12 pp merged or unmerged — C2 paid a page in the section-drop plan only because
> that plan sat just the wrong side of a boundary; **C2 is off the critical path and should
> now be drawn for the reader, not the page count**). Three §2 moves + claim 21's power
> numbers + §8's derivations + demoted Related Work + two banked appendices + a merged
> figure = 1 page. **The paper's length is its results, not its packaging** — a fourth
> independent route to the same answer. At 12 pp it is not workshop-shaped on any CFP read
> (limits 5/8/9), and not by a margin packaging reaches. Remaining options: a venue whose
> limit fits, a length exception, or compressing §5/§6 — the last is the only *drafting*
> decision left and the only one that risks claims, so cost it with `move_cost.py` first.
> **Note: no measured compression of §5/§6 exists** — 1.85/0.65 are rebalance *budgets*,
> never applied edits (§5 measures 2.13 today, §6 0.90).
>
> One content fix the demotion needs: `cut_dependencies.RW_PARAGRAPH` was drafted for the
> plan that *deleted* §5, so it drops the circuit-analysis citations. §5 stays in both
> versions ⇒ they must be added back (`two_versions.CIRCUIT_CLAUSE`).
>
> Historical, for the record — **the leading plan was 9 pp in ACL and 10 in NeurIPS, +1 in both**
> (`logs/plan_compile.md`, `scripts/plan_compile.py`). The plan's edits are imported from
> `move_cost.py` and `cut_dependencies.py` rather than re-typed, so the compiled paper and
> the costed plan cannot drift apart. Superseded reasoning, kept because it is the kind
> that does not announce itself: the estimate was 8.46 + a **carried 0.48-page offset**
> ≈ 8.9, and it landed right — but the 0.48 was the estimator's residual on the *full*
> draft, and that residual is a property of the text (float placement, multi-line table
> cells), not of the estimator, so it had no business being carried onto a plan that
> deletes §5 and §6. It also dressed a whole-page overflow as 0.9 of one. Page counts are
> integers.
>
> **THE FIGURE ALLOWANCE WAS NEVER MEASURED, AND IT IS THE WHOLE REMAINING GAP.**
> `FIGURE_LINES` charges 38 slots per figure — a budget written before the figures were
> drawn. The figures are 7.1 × 2.45 in and 7.1 × 2.70 in against a text block measured at
> 455.2 pt, so their aspect ratio fixes their cost: **27 and 29 slots, not 38 and 38.**
> More important than the 0.19-page over-charge is what a *merge* is worth, priced as a
> curve of real compiles rather than a point:
>
> | figure treatment | slots | compiled |
> |:--|--:|--:|
> | no figures at all (floor, not a proposal) | 0 | 8 pp |
> | **C2 one figure, redesigned** | 30 | **8 pp** |
> | C1 one figure, stacked under a shared caption | 53 | 9 pp |
> | C0 two figures, at their real aspect ratio | 57 | 9 pp |
> | two figures at the assumed budget | 76 | 9 pp |
>
> **C2 closes the last page; §3 and §4 prose need not be touched at all.** The merge that
> pays is not a layout trick — figure 2's early and delayed arms are the same `EARLY` and
> `DELAY` numbers figure 1 already plots, so folding its treated arm into figure 1's
> per-size panels duplicates nothing and loses nothing. C1 (stack, share a caption) buys
> one caption and does not move the compile: **do not attempt it.**
>
> **The two formats stop being length-neutral here, and the figure spec differs by venue.**
> Length-neutrality held while both were failing by five pages; at the margin it breaks,
> because NeurIPS is single-column and a float displaces text it cannot pack around.
> Bisected turnovers: **BabyLM/ACL 43 slots, interpscience/NeurIPS 24.**
>
> | format | limit | turnover | C2 as sized (30 slots) |
> |:--|--:|--:|--:|
> | ACL / BabyLM | 8 | 43 slots | 8 pp — **fits** |
> | NeurIPS / interpscience | 9 | 24 slots | 10 pp (+1) |
>
> So C2 clears BabyLM by 13 slots but is **7 slots over for interpscience**: at 7.1 in wide
> the merged figure may be 4.17 in tall for BabyLM and only **2.11 in** for interpscience —
> **shorter than figure 1 alone is today (2.45 in)**. That is the real design constraint and
> it is not obvious. Either way **no §3/§4 prose cut is required**; interpscience needs the
> figure drawn smaller, not the argument cut. C2's 30 slots is the one modelled number in
> the chain — **draw the figure and re-run `plan_compile.py` before relying on any of it.**
>
> **The CFP appendix question is now BLOCKING, not a nicety** (`logs/plan_compile.md` §4).
> The plan reaches 8 pp partly by moving §2 material into an appendix; compiled with that
> reversed it is **9 pp**. At a five-page overflow the answer changed nothing, so it was
> correctly deprioritised — its value rose while it sat in the queue, as the overflow shrank
> underneath it. `notes/interpscience-format-query.md`.
>
> Order that follows: **answer the CFP question and pick the venue, then draw the figure,
> then prose — if prose is still needed at all.** Prose cuts are irreversible in review and
> a figure redesign is not, so spending §3/§4 substance on a page the layout can supply is
> the one sequencing error worth avoiding here.

**Related Work placement.** It sits *before* §6 and after §5, so the novelty argument lands
while the core results are fresh and before the bonus cluster. Keeping it inside the
Discussion was the alternative — it saves ~0.15 page of heading overhead but puts the
positioning after every result, where a skeptical reader has already formed a view.

---

## §1 Introduction

- Critical periods are studied in models whose pretraining history cannot be controlled;
  the contribution is a design where *when* a skill enters the stream is the manipulated
  variable and total exposure is held fixed.
- Contributions in body order: (i) a replicated arithmetic critical period at two sizes;
  (ii) an absolute skill dissociation, stated with its scope; (iii) an intervention that
  reopens the deficit, and a mechanism attribution; (iv) a single-method circuit read-out.
- **Name predictions #1–#3 here as predictions**, pointing to §7.2 for outcomes. Do not
  report outcomes here; §11 of the backbone fixes their location.
- No claim rows land in this section.

## §2 Methods

- **2.1 Models and corpus.** 11M / 23M / 60M from random init, BabyLM strict-small, custom
  BPE, RoPE, pre-norm, GELU, untied embeddings, AdamW + cosine. No pretrained checkpoints
  anywhere, baselines included — state as the design constraint that makes the timing
  manipulation meaningful, not as a detail.
- **2.2 Delay schedule and exposure matching.** Onset as a fraction of total steps;
  exposure matched by construction; run_id / checkpoint convention.
- **2.3 Measures (backbone §5.4, verbatim obligations).** Arithmetic = teacher-forced exact
  match at step 5000 on 500 held-out problems; syntax = continuation-scored licensing
  interaction; general capability = held-out val perplexity as the three-number
  decomposition. The licensing-interaction calibration belongs here — raw BLiMP pairs are
  dominated by the "that" unigram prior, and a frequency-only model sits at exactly 0.500.
  It is why the syntax numbers mean anything, so it is a result, not a note.
- **2.4 Eval-set hygiene.** One held-out set; other draws overlap training by ~17–19 %; the
  contamination was found and corrected mid-project; cross-harness floor ~0.005, so
  accuracies compare down to ~0.01 and no finer. **Report the correction, not only the
  corrected numbers.**
- **2.5 The 11M floor exclusion.** Pre-registered. Required wording: "excluded by the
  pre-registered floor: a capacity/budget boundary" — never "no critical period at 11M".
- **2.6 Retired analysis designs.** All three pre-registered designs were un-estimable on
  the surviving design; state what was planned, why it was not run, what replaced it.
- **2.7 Multiplicity.** Families F1–F6 with their policies; exploratory tests flagged and
  barred from headlines.

## §3 Detect

### 3.1 The arithmetic critical period (C1)

| carries | label |
|:--|:--|
| **1** deficit 0.214 ± 0.055 (23M) and 0.229 ± 0.037 (60M), same direction every seed | ESTABLISHED |
| **2** similar magnitude at both sizes | ESTABLISHED — travels with MDE 0.144; never "size does not matter" |
| **3** severity is seed-variable, driven by the delayed endpoint | ESTABLISHED |

- Per-seed paired deficits with per-seed values printed; Holm within F1 (0.0107 / 0.0087).
- Claim 3 is why the paper reports absolute endpoints over percent-of-gap — make the link
  here so §4 need not re-argue it.
- **Forward-reference to §5's size null.** Backbone §2.1 requires C1's well-powered null
  (MDE 0.144 < the 0.21 deficit) and C4's underpowered one (MDE 0.809 > the 0.681 span) to
  be written as a matched pair. §3.1 opens the pair; **§5 must carry the back-reference** —
  if either half ships alone the reader over-trusts one and under-trusts the other.
- **§8 items placed here:** the 11M paired deficit p = 0.0345 (number, exclusion, reason,
  in that order); the d25 / d55 / d75 dose points (n = 1, plotted with no fitted trend).

### 3.2 Long-distance *wh*-licensing shows no detectable critical period (C3)

| carries | label |
|:--|:--|
| **11** no detectable LD critical period under five distinct probes | ESTABLISHED (null, scoped) |
| **12** structure is encoded, its readout is timing-invariant (positive control passes) | ESTABLISHED |

- Five probes: endpoint, dynamics, rate, representational, LR-budget. **Not five tests of
  one hypothesis** — write "five distinct routes to the same negative, each with its own
  detectable-effect floor", never "confirmed five ways".
- The size-sweep gate (F4) DISCONFIRM sits here with its threshold sensitivity: it is why
  syntax was dropped from the grid.

### 3.3 Skill dissociation (C1 × C3)

| carries | label |
|:--|:--|
| **11b** arithmetic pays 0.214 ± 0.055; LD pays nothing measurable across five probes | ESTABLISHED **(scoped)** |

- Scope stated in place, as a table: syntax's 0.0283 headroom vs arithmetic's 0.791, and
  LD MDE 0.048 > total headroom. Absolute dissociation established; differential
  **susceptibility** not — unmeasurable at achieved n under the mean readout, **un-run**
  under the rate readout.
- **Backbone §10 case-(c) statement, required:** the master plan anticipated a
  *reopening*-side dissociation; the data support a *detection*-side one. Say they are not
  the same claim.
- Forward-reference the 14-seed 60M rate-collapse test to §8 as the specific experiment
  that decides the susceptibility half — never as "future work could investigate".

## §4 Reopen — the LR-shift intervention (C2)

| carries | label |
|:--|:--|
| **4** endpoints 0.767 ± 0.014 / 0.796 ± 0.022 **beside** nulls 0.577 ± 0.051 / 0.600 ± 0.040 | ESTABLISHED |
| **5** recovered endpoint scale-invariant (both inside the pre-registered [0.722, 0.812]) | ESTABLISHED |
| **6** intervention collapses cross-seed endpoint variance | ESTABLISHED |
| **7** closure at d50 is largely LR-schedule, not entrenchment (cram+lr_shift ≈ lr_shift ≫ cram, n = 3) | ESTABLISHED |
| **9** reset / spike / cram show no detectable recovery; freeze is harmful | SUGGESTIVE (n = 1) |
| **10** 60M general-LM cost bounded above by 1.67 ppl, likely a horizon artifact | BOUNDED-NOT-EXCLUDED |
| **8** ~70 % of the 11M effect requires the delayed exposure profile (H3) | SUGGESTIVE (exploratory) |

- Order: gate on absolute endpoints (F3) → per-seed lifts (F2, Holm 0.0065 / 0.0054) →
  perplexity accounting → negative control → cram decomposition yielding claim 7.
- **The forgetting pitfall is discharged here**, not deferred: recovery and retention
  together at every point, and the 60M cost quoted *with* its bound and the matched-LR
  reasoning. A bounded cost stated bare reads as a hidden failure.
- Claim 9 is *measured* here; prediction #2's standalone sentence is §7.2. In this section
  only: "no detectable recovery at n = 1 (inside the null's cross-seed range)".
- Claim 5 belongs in this section's lede as well as the abstract's final clause.
- **§8 item placed here:** the 11M gate PASS (degenerate — the arm exceeds its own anchor).

## §5 Circuit identity (C4)

| carries | label |
|:--|:--|
| **13** recovered models range from repair to reorganization | ESTABLISHED (single-method) |
| **14** 60M reorganizes at every seed | ESTABLISHED (single-method) |
| **16b** within the shallow-deficit regime the deeper seed shows *more* repair | ESTABLISHED (single-method; disjoint CIs, common held-out draw) |
| **16** 60M identity more reproducible than 23M (SD 2.7×) | SUGGESTIVE |
| **15** deficit depth grades identity | BOUNDED-NOT-EXCLUDED (ρ = −0.71, exact p = 0.068, LOO-fragile) |

- **Earn the single-method annotation in-text:** mean-ablation discriminates; whole-layer
  ablation saturates; CKA is flat (0.66–0.82 regardless); single-head patching is flat
  (r = 0.045 with the primary, its corroboration retracted); pruning pre-screened NO-GO
  (sign is hyperparameter-set). Three non-discriminating methods reported as such is what
  licenses the annotation.
- **Back-reference to §3.1** — the matched-pair obligation on the two size nulls.
- Every bootstrap CI here is eval-item variance only, n = 1 model per point — label at
  each appearance.
- **§8 items placed here:** scale as an independent driver (estimable, underpowered —
  p = 0.39, MDE 0.809 vs a 0.681 span → *untested*); the within-60M Spearman; the 23M-only
  extrapolation; anything from `week8_causal_trace.py` / `week8_pruning_feasibility.py`;
  circuit redundancy / sufficiency.

## §6 Agreement attraction (C5) — compressed section

| carries | label |
|:--|:--|
| **17** attraction is real, not a frequency artifact | ESTABLISHED |
| **18** attraction declines with scale | ESTABLISHED (syntax) / SUGGESTIVE (arithmetic) |
| **19** attraction dissociates from general complexity | ESTABLISHED (single-mixture) + BOUNDED-NOT-EXCLUDED (replication) |
| **20** below-chance agreement only at the smallest size | ESTABLISHED |
| **21** attraction declines *within* training (steps 2500 → 5000) | SUGGESTIVE (exploratory) — added 2026-08-05; power numbers travel with it |

- Source: `notes/agreement-attraction-writeup.md` §§1–8, compressed to ~0.65 page.
- **Compression is label-preserving, not prose-shortening.** Claim 18 quotes "p = 0.033
  OLS / 0.073 robust"; claim 19's replication half is "does not survive replication; not
  excluded (CI contains +0.133)". If a sentence cannot carry its label at length, it is cut
  entirely rather than trimmed into an unlabelled version.
- Keep the methodological byproduct (three metrics, three conclusions; the pooled-72
  averaging trap) — it is the part that earns the section its place beside the CP work.

## §7 Discussion

- **7.1 The linking sentence (required).** Both results are timing effects at different
  levels: 11b says *which* skills a late start damages is skill-specific; 7 says the damage
  is carried by the optimizer's schedule rather than by representations hardened against
  revision.
- **7.2 Prediction accountability — three standalone sentences, each naming its prediction
  and its outcome. Non-negotiable.**
  - **#1 syntax stronger → INVERTED.** Arithmetic carries the effect; LD shows none
    detectable under five probes. Origin of 11b, not an aside.
  - **#2 reset most effective → NOT BORNE OUT.** No detectable recovery (0.586, inside the
    null's range [0.524, 0.626]); `lr_shift` alone cleared the gate. SUGGESTIVE (n = 1),
    never "reset fails". The forgetting-risk half was also not observed as predicted.
  - **#3 reset ⇒ new circuit → UNADDRESSED, antecedent never occurred.** Reset never
    recovered, so the conditional was never entered; §5 diagnoses `lr_shift`-recovered
    models, which #3 did not cover. **Name the mismatch** — claims 13–15 sit exactly where
    a reader expects #3's answer.
- **7.3 Scope of the mechanism finding.** Schedule-driven closure is demonstrated at d50,
  for arithmetic, at two sizes. Not a claim that entrenchment plays no role at other onsets.
- ~~7.4 Relation to prior work~~ — **moved out 2026-08-06** to the standalone Related Work
  section between §5 and §6. §7 ends at 7.3. One line of C5 connective tissue is still owed
  to the Discussion (see the note at the end of `draft-section-6.md`) and has no home yet.

## §8 Limitations

Backbone §9's label-movers, each as a specific experiment:

1. **14 delayed 60M syntax seeds** (rate-collapse at thr 0.5593 against the existing 24-seed
   d0 baseline; 6 at the permissive 0.55) — cheapest outstanding test, and the only one that
   promotes 11b past ESTABLISHED (scoped). Assumes delay eliminates the high mode; if it
   halves it, 59. At 23M the same test needs 82 and is out of reach.
2. **60M arm-extension past step 5000** → converts claim 10's bound to a point estimate.
3. **More triads at either size** → the only thing that moves claim 15; at n = 6 with
   ρ = −0.71 the question is open, not answered.
4. **Joint head-group patching with demonstrated dynamic range** → would drop 13–15's
   single-method annotation.
5. **3+ further 11.5M SVA seeds** → moves claim 18 (arithmetic) if the robust p drops with
   leverage rebalanced.
- Also: single corpus, single tokenizer, ≤ 60M, one delay onset carrying the reopening
  result, 11M excluded by floor.

## §9 Conclusion

No new claims. The two-level timing statement from 7.1, and the one open question worth
naming (susceptibility vs absolute dissociation).

---

## Coverage check — every ledger row has exactly one home

| claim | home | in abstract? |
|--:|:--|:--|
| 1 | §3.1 | yes (anchor) |
| 2 | §3.1 | no |
| 3 | §3.1 | no — recoverable from the ± SDs |
| 4 | §4 | yes (anchor) |
| 5 | §4 lede | yes, final clause |
| 6 | §4 | yes (cut priority 1) |
| 7 | §4 | yes (anchor) |
| 8 | §4 (exploratory, flagged) | no |
| 9 | §4 measured, §7.2 stated | no — must not drift in |
| 10 | §4 | yes, with its bound |
| 11 | §3.2 | yes |
| 11b | §3.3 | yes (anchor, scoped in-abstract) |
| 12 | §3.2 | yes — the probe-4 clause, never cut |
| 13 · 14 · 15 · 16 · 16b | §5 | no |
| 17 · 18 · 19 · 20 · 21 | §6 | no |

**§8 NOT-A-CLAIM placement:** 11M Detect p and the dose points → §3.1; 11M Reopen gate →
§4; within-60M Spearman, scale-as-driver, 23M extrapolation, causal-trace/pruning numbers,
redundancy → §5; "±0.02 run noise" → nowhere, it is not a threshold anywhere.

**Deliberately absent from the whole paper:** any recovery-percentage headline; any
cross-method-validation language for C4; any reopening-side dissociation claim; the retired
"20×" ratio; any bare null without its MDE.

---

## Venue landscape — searched 2026-08-05

| route | date | status |
|:--|:--|:--|
| BabyLM 2026 direct | July 15, 2026 | **closed** (user-verified) |
| BabyLM 2026 via ARR commitment | May 25, 2026 | **closed** (user-verified) |
| **NeurIPS 2026 workshops** | contributions **Aug 29, 2026** (NeurIPS-suggested date; each workshop sets its own) | **OPEN — 24 days out** |
| ICLR 2027 main conference | abstract Sept 19 / paper Sept 24, 2026 | open, but a main-track submission is a different paper |
| ICLR 2027 workshops | ~Feb 2027 | opens after the ICLR 2027 CFP cycle |
| BabyLM 2027 | ~mid-July 2027 (if the 2026 pattern holds) | the format target |

**The near-term option is real but tight.** Accepted NeurIPS 2026 workshops were announced
July 11; the suggested contribution deadline is August 29. Best thematic fit found is
**"Interpretability as a Science"** (NeurIPS 2026, Sydney) — explicitly about grounding
interpretability as a rigorous empirical science, with causal claims, falsifiability and
measurement standards as named topics. That is unusually close to this project's actual
methodological spine (MDE-over-p, demonstrated discrimination before claiming corroboration,
one label per claim). Also live: *Interpretability for Discovery*, *XAI4Science*. Not a fit:
*NeurReps*. ICML 2026's mech-interp workshop has passed.

**RESOLVED 2026-08-09 — the three CFPs were read, and all three are non-archival:**

| workshop | limit | deadline | format | archival |
|:--|:--|:--|:--|:--|
| **Interpretability as a Science** | short 5 / **long 9** | **Aug 28, 2026** | ICLR *or* NeurIPS to submit; NeurIPS for camera-ready | non-archival |
| XAI4Science | regular 8 / tiny 5, excl. references **and appendices** | Aug 29, 2026 AoE | NeurIPS 2026 template | non-archival, no proceedings |
| Interpretability for Discovery | 5 (+1 for camera-ready) | Aug 29, 2026 AoE | NeurIPS 2026 workshop template | non-archival |

Interpscience is both the best thematic fit and the most generous limit, and its CFP page
does now resolve (`interpscience.github.io/cfp.html`). Non-archival across the board, which
is what keeps the arXiv → BabyLM 2027 path open rather than substituting for it.

**Two things the CFPs do NOT say, both worth one email before relying on them:**
interpscience is silent on whether references count toward the 9 pages and on whether
appendices are permitted (XAI4Science is explicit that both are excluded; interpscience is
not). Neither uncertainty is load-bearing at a 5-page gap.

**Decision impact: none on format, and none on length.** See the ⚠ block above — the
formats are length-neutral (13 ACL against 8, 14 NeurIPS against 9). A workshop version
would be a compression of the same draft by the same amount, not an escape from it.
The remaining reason to choose a workshop is fit and calendar, not pages.

## Open items

- **Confirm the interpscience (or alternative NeurIPS-workshop) CFP by hand** if the
  near-term route is wanted — 24 days is enough for a workshop-length compression of §§3–4
  plus the abstract, but not for §§5–6 to reach the same polish.
- ~~Data-lineage fix in `seeds23_replication_results.md`~~ — **APPLIED 2026-08-05.** Seed 1's
  ppl cells were mixed-lineage (canonical accuracy 0.524 paired with the superseded run's
  50.83). Corrected to null **50.747** / arm **49.858** / c − b **−0.889**, with the
  superseded values retained in a dated correction note and the summary bullet updated to
  the three-seed set {−0.89, −0.20, +0.20}. No conclusion moved; the 23M no-breach finding
  strengthens slightly (two of three seeds improve).
