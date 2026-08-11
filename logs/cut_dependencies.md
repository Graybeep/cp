# Dropping §5: what breaks, what it frees, and what to do about Related Work

Source: `scripts/cut_dependencies.py`. Line deltas measured with `page_estimate`'s wrap,
same as `move_cost.py`.

## Part 1 — every inbound reference to §5

| kind | where | what depends on §5 | resolving it |
|:--|:--|:--|--:|
| **ORPHAN** | §1 | the MDE reporting rule's worked example points at §5.3 | +2 ln freed |
| **FREED** | §2.4 | the eval-contamination correction exists only because it hit §5's numbers | +8 ln freed |
| **DISSOLVED** | §3.1 | the backbone §5.5 paired-null dependency | +5 ln freed |
| **ORPHAN** | §7.2 | prediction #3's accountability statement is written around §5's position | +5 ln freed |
| **FREED** | §8 | two of the five costed label-movers move labels that would no longer exist | +12 ln freed |
| **FREED** | Related Work | the circuit-identity citation paragraph positions §5 and nothing else | +8 ln freed |

- **§1 (ORPHAN)** — The rule survives — §3.1 and §3.3 still use it. Only the example dies.
- **§2.4 (FREED)** — The harness-agreement floor (~0.005) stays: §3 and §4 compare across harnesses too.
- **§3.1 (DISSOLVED)** — NOT a violation. §5.5 forbids either size null shipping alone; the underpowered one IS §5's, so removing §5 removes both halves and the rule is satisfied vacuously. Claim 2 keeps its own MDE and needs no partner.
- **§7.2 (ORPHAN)** — The §11 obligation to account for #3 survives and gets simpler: with no circuit section there is no longer a place a reader mistakes for its answer.
- **§8 (FREED)** — §8 opens "Five specific experiments" — becomes three. One-word edit, not costed here.
- **Related Work (FREED)** — Also retires four unverified citations, which is a small risk reduction as well as a saving.

**§5 drags 40 further lines out with it** (0.39 page) beyond the section itself: 33 lines of material that exists only to support it, plus 7 from shortening the two passages that merely point at it.

**The abstract needs no edit at all** — it contains no circuit content (claims 13–16 were
never abstract anchors). That is the strongest single indication that §5 is genuinely
severable rather than load-bearing.

**And the one rule that looked like a blocker is not one.** Backbone §5.5 forbids claim 2's
well-powered size null shipping without its underpowered twin. The twin is §5's, so deleting
§5 removes both halves and satisfies the rule by absence rather than breaking it. Worth
stating explicitly in the §5.5 record, because the next reader of that rule will assume the
opposite.

## Part 2 — Related Work: demote rather than delete

The standalone section is **78 lines (0.77 page)**. Deleting it outright buys all of that and
discards the positioning argument. Demoting it to a single Discussion paragraph that keeps
the plasticity/warm-starting distinction near full length, compresses Achille and grokking
to a clause each, and pushes the BabyLM/BLiMP/agreement citations inline to §2.3 and §6,
costs **27 lines** — so it buys **51 lines (0.50 page)** against deletion's 78.

**The positioning function therefore costs 27 lines (0.26 page) to keep, not 78.** That is the
whole argument for demoting: the section's own length note calls the plasticity paragraph
the place "where a reader decides whether the schedule finding is a rediscovery", and a
paper that drops it saves a third of a page by leaving its most likely reviewer objection
unanswered. Note also that the circuit-identity citations leave anyway with §5 (Part 1),
so the demoted paragraph is shorter than the section minus that paragraph would suggest.

Draft of the replacement paragraph is in this script (`RW_PARAGRAPH`) so the measurement
and the text cannot drift apart.
